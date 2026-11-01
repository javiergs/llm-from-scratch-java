package com.javiergs.llm.model;

import com.javiergs.llm.tokenizer.CharacterTokenizer;
import java.util.Arrays;
import org.nd4j.linalg.api.ndarray.INDArray;

/**
 * Demonstrates a full untrained GPT-style forward pass.
 *
 * @author Javier Gonzalez-Sanchez
 * @version 0.1.0 (2026-10-31)
 */
public final class GptModelExample {
    private GptModelExample() {
    }

    /**
     * Prints the vocabulary logits for each position in a short prompt.
     *
     * @param args unused command-line arguments
     */
    public static void main(String[] args) {
        String prompt = "Hello!";
        CharacterTokenizer tokenizer = new CharacterTokenizer(prompt);
        int[] ids = tokenizer.encode(prompt);
        GptModel model = new GptModel(tokenizer.vocabularySize(), 16, 8, 2, 2, 32, 42L);
        INDArray logits = model.forward(ids);
        System.out.println("Token IDs: " + Arrays.toString(ids));
        System.out.println("Logits shape: " + Arrays.toString(logits.shape()));
        System.out.println("Last-position logits: " + logits.getRow(ids.length - 1));
        System.out.println("These scores are random until the model is trained.");
    }
}
