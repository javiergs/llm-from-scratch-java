package com.javiergs.llm.transformer;

import com.javiergs.llm.embedding.PositionalEmbedding;
import com.javiergs.llm.embedding.TokenEmbedding;
import com.javiergs.llm.tokenizer.CharacterTokenizer;
import java.util.Arrays;
import org.nd4j.linalg.api.ndarray.INDArray;

/**
 * Demonstrates a complete forward pass through one transformer block.
 *
 * @author Javier Gonzalez-Sanchez
 * @version 0.1.0 (2026-10-31)
 */
public final class TransformerBlockExample {
    private TransformerBlockExample() {
    }

    /**
     * Builds embeddings and passes them through a pre-norm transformer block.
     *
     * @param args unused command-line arguments
     */
    public static void main(String[] args) {
        String text = "Hello!";
        CharacterTokenizer tokenizer = new CharacterTokenizer(text);
        INDArray tokens = new TokenEmbedding(tokenizer.vocabularySize(), 4, 42L)
                .forward(tokenizer.encode(text));
        INDArray input = new PositionalEmbedding(16, 4, 43L).addTo(tokens);
        INDArray output = new TransformerBlock(4, 2, 16, 44L).forward(input);
        System.out.println("Input shape: " + Arrays.toString(input.shape()));
        System.out.println("Output shape: " + Arrays.toString(output.shape()));
        System.out.println("Transformer output:\n" + output);
    }
}
