package com.javiergs.llm.attention;

import com.javiergs.llm.embedding.PositionalEmbedding;
import com.javiergs.llm.embedding.TokenEmbedding;
import com.javiergs.llm.tokenizer.CharacterTokenizer;
import java.util.Arrays;
import org.nd4j.linalg.api.ndarray.INDArray;

/**
 * Demonstrates two causal attention heads processing the same sequence.
 *
 * @author Javier Gonzalez-Sanchez
 * @version 0.1.0 (2026-10-31)
 */
public final class MultiHeadAttentionExample {
    private MultiHeadAttentionExample() {
    }

    /**
     * Prints per-head attention probabilities and the output dimensions.
     *
     * @param args unused command-line arguments
     */
    public static void main(String[] args) {
        String text = "Hello!";
        CharacterTokenizer tokenizer = new CharacterTokenizer(text);
        int[] ids = tokenizer.encode(text);
        INDArray tokens = new TokenEmbedding(tokenizer.vocabularySize(), 4, 42L).forward(ids);
        INDArray input = new PositionalEmbedding(16, 4, 43L).addTo(tokens);
        MultiHeadAttention attention = new MultiHeadAttention(4, 2, 44L);
        for (int head = 0; head < 2; head++) {
            System.out.println("Head " + head + " probabilities:\n" + attention.attentionWeights(input, head));
        }
        System.out.println("Output shape: " + Arrays.toString(attention.forward(input).shape()));
    }
}
