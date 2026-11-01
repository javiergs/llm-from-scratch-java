package com.javiergs.llm.attention;

import com.javiergs.llm.embedding.PositionalEmbedding;
import com.javiergs.llm.embedding.TokenEmbedding;
import com.javiergs.llm.tokenizer.CharacterTokenizer;
import java.util.Arrays;
import org.nd4j.linalg.api.ndarray.INDArray;

/**
 * Demonstrates causal self-attention after token and position embeddings.
 *
 * @author Javier Gonzalez-Sanchez
 * @version 0.1.0 (2026-10-31)
 */
public final class AttentionExample {
    private AttentionExample() {
    }

    /**
     * Prints the lower-triangular attention matrix and contextualized vectors.
     *
     * @param args unused command-line arguments
     */
    public static void main(String[] args) {
        String text = "Hello!";
        CharacterTokenizer tokenizer = new CharacterTokenizer(text);
        int[] ids = tokenizer.encode(text);
        INDArray tokens = new TokenEmbedding(tokenizer.vocabularySize(), 4, 42L).forward(ids);
        INDArray input = new PositionalEmbedding(16, 4, 43L).addTo(tokens);
        CausalSelfAttention attention = new CausalSelfAttention(4, 44L);
        System.out.println("Token IDs: " + Arrays.toString(ids));
        System.out.println("Attention probabilities:\n" + attention.attentionWeights(input));
        System.out.println("Output shape: " + Arrays.toString(attention.forward(input).shape()));
    }
}
