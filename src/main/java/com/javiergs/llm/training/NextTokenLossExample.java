package com.javiergs.llm.training;

import com.javiergs.llm.model.GptModel;
import com.javiergs.llm.tokenizer.CharacterTokenizer;
import org.nd4j.linalg.api.ndarray.INDArray;

/**
 * Demonstrates next-token loss for a randomly initialized GPT-style model.
 *
 * @author Javier Gonzalez-Sanchez
 * @version 0.1.0 (2026-10-31)
 */
public final class NextTokenLossExample {
    private NextTokenLossExample() {
    }

    /**
     * Computes loss without changing model weights.
     *
     * @param args unused command-line arguments
     */
    public static void main(String[] args) {
        String text = "hello world!";
        CharacterTokenizer tokenizer = new CharacterTokenizer(text);
        int[] ids = tokenizer.encode(text);
        GptModel model = new GptModel(tokenizer.vocabularySize(), 32, 8, 2, 2, 32, 42L);
        INDArray logits = model.forward(ids);
        System.out.printf("Mean next-token cross-entropy: %.4f%n", NextTokenLoss.mean(logits, ids));
        System.out.println("This measures prediction error; no training has occurred.");
    }
}
