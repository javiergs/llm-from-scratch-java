package com.javiergs.llm.embedding;

import com.javiergs.llm.tokenizer.CharacterTokenizer;
import org.nd4j.linalg.api.ndarray.INDArray;

/**
 * Demonstrates converting text into ND4J token embedding vectors.
 *
 * @author Javier Gonzalez-Sanchez
 * @version 0.1.0 (2026-10-31)
 */
public final class EmbeddingExample {
    private EmbeddingExample() {
    }

    /**
     * Tokenizes a short sentence and prints its embedding matrix.
     *
     * @param args unused command-line arguments
     */
    public static void main(String[] args) {
        String text = "Hello LLM!";
        CharacterTokenizer tokenizer = new CharacterTokenizer(text);
        TokenEmbedding embedding = new TokenEmbedding(tokenizer.vocabularySize(), 4, 42L);
        INDArray vectors = embedding.forward(tokenizer.encode(text));
        System.out.println("Embedding shape: " + java.util.Arrays.toString(vectors.shape()));
        System.out.println(vectors);
    }
}
