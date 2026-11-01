package com.javiergs.llm.embedding;

import com.javiergs.llm.tokenizer.CharacterTokenizer;
import java.util.Arrays;
import org.nd4j.linalg.api.ndarray.INDArray;

/**
 * Demonstrates combining token identity and absolute position vectors.
 *
 * @author Javier Gonzalez-Sanchez
 * @version 0.1.0 (2026-10-31)
 */
public final class PositionalEmbeddingExample {
    private PositionalEmbeddingExample() {
    }

    /**
     * Encodes a short text and prints token, position, and combined shapes.
     *
     * @param args unused command-line arguments
     */
    public static void main(String[] args) {
        String text = "Hello LLM!";
        CharacterTokenizer tokenizer = new CharacterTokenizer(text);
        int[] ids = tokenizer.encode(text);
        TokenEmbedding tokenEmbedding = new TokenEmbedding(tokenizer.vocabularySize(), 4, 42L);
        PositionalEmbedding positionalEmbedding = new PositionalEmbedding(32, 4, 43L);

        INDArray tokens = tokenEmbedding.forward(ids);
        INDArray positions = positionalEmbedding.forward(ids.length);
        INDArray combined = positionalEmbedding.addTo(tokens);

        System.out.println("Token IDs: " + Arrays.toString(ids));
        System.out.println("Token vectors: " + Arrays.toString(tokens.shape()));
        System.out.println("Position vectors: " + Arrays.toString(positions.shape()));
        System.out.println("Combined vectors: " + Arrays.toString(combined.shape()));
        System.out.println("Combined embeddings:\n" + combined);
    }
}
