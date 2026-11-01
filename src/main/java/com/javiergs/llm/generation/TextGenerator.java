package com.javiergs.llm.generation;

import com.javiergs.llm.model.GptModel;
import com.javiergs.llm.tokenizer.CharacterTokenizer;
import java.util.Arrays;
import org.nd4j.linalg.api.ndarray.INDArray;

/**
 * Generates characters autoregressively by repeatedly predicting the next token.
 * Uses deterministic greedy decoding for reproducible educational examples.
 *
 * @author Javier Gonzalez-Sanchez
 * @version 0.1.0 (2026-10-31)
 */
public final class TextGenerator {
    private TextGenerator() {
    }

    /**
     * Extends a nonempty token prompt using greedy next-token predictions.
     * When the sequence exceeds the model context, only the most recent
     * contextLength tokens are used for the next prediction.
     *
     * @param model trained or untrained GPT model
     * @param prompt nonempty starting token IDs
     * @param newTokens number of tokens to append, zero or greater
     * @param contextLength model context window size
     * @return prompt followed by generated token IDs
     */
    public static int[] generate(GptModel model, int[] prompt, int newTokens, int contextLength) {
        if (model == null || prompt == null) {
            throw new NullPointerException("Model and prompt must not be null");
        }
        if (prompt.length == 0 || newTokens < 0 || contextLength <= 0) {
            throw new IllegalArgumentException("Expected a nonempty prompt, nonnegative steps, positive context");
        }
        if (newTokens > Integer.MAX_VALUE - prompt.length) {
            throw new IllegalArgumentException("Requested output is too long");
        }
        int[] output = Arrays.copyOf(prompt, prompt.length + newTokens);
        // Validate prompt tokens even when no new tokens are requested.
        for (int id : prompt) {
            if (id < 0 || id >= model.vocabularySize()) {
                throw new IllegalArgumentException("Invalid prompt token ID: " + id);
            }
        }
        for (int step = 0; step < newTokens; step++) {
            int end = prompt.length + step;
            int start = Math.max(0, end - contextLength);
            int[] context = Arrays.copyOfRange(output, start, end);
            INDArray logits = model.forward(context);
            int row = Math.toIntExact(logits.size(0)) - 1;
            int next = 0;
            double maximum = Double.NEGATIVE_INFINITY;
            for (int token = 0; token < model.vocabularySize(); token++) {
                double score = logits.getDouble(row, token);
                if (!Double.isFinite(score)) {
                    throw new IllegalStateException("Model produced non-finite logits");
                }
                if (score > maximum) {
                    maximum = score;
                    next = token;
                }
            }
            output[end] = next;
        }
        return output;
    }

    /**
     * Encodes a prompt, generates additional characters, and decodes the result.
     *
     * @param model model whose vocabulary matches the tokenizer
     * @param tokenizer character-level tokenizer
     * @param prompt nonempty text using known characters
     * @param newTokens number of characters to generate
     * @param contextLength model context length
     * @return prompt and generated characters
     */
    public static String generateText(GptModel model, CharacterTokenizer tokenizer,
                                      String prompt, int newTokens, int contextLength) {
        if (model == null || tokenizer == null || prompt == null) {
            throw new NullPointerException("Model, tokenizer, and prompt must not be null");
        }
        if (tokenizer.vocabularySize() != model.vocabularySize()) {
            throw new IllegalArgumentException("Tokenizer vocabulary must match the model");
        }
        return tokenizer.decode(generate(model, tokenizer.encode(prompt), newTokens, contextLength));
    }
}
