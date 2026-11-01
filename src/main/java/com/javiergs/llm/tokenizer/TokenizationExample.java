package com.javiergs.llm.tokenizer;

import java.util.Arrays;

/**
 * Demonstrates reversible character-level tokenization.
 *
 * @author Javier Gonzalez-Sanchez
 * @version 0.1.0 (2026-10-31)
 */
public final class TokenizationExample {
    private TokenizationExample() {
    }

    /**
     * Prints a sample vocabulary, token IDs, and decoded text.
     *
     * @param args unused command-line arguments
     */
    public static void main(String[] args) {
        String text = "Hello LLM!";
        CharacterTokenizer tokenizer = new CharacterTokenizer(text);
        int[] tokens = tokenizer.encode(text);

        System.out.println("Vocabulary size: " + tokenizer.vocabularySize());
        System.out.println("Token IDs: " + Arrays.toString(tokens));
        System.out.println("Decoded text: " + tokenizer.decode(tokens));
    }
}
