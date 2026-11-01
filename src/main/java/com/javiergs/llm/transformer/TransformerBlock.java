package com.javiergs.llm.transformer;

import com.javiergs.llm.attention.MultiHeadAttention;
import org.nd4j.linalg.api.ndarray.INDArray;

/**
 * GPT-style pre-normalization transformer block.
 * Applies causal attention and a feed-forward network with residual connections.
 *
 * @author Javier Gonzalez-Sanchez
 * @version 0.1.0 (2026-10-31)
 */
public final class TransformerBlock {
    private final LayerNormalization attentionNormalization;
    private final LayerNormalization feedForwardNormalization;
    private final MultiHeadAttention attention;
    private final FeedForward feedForward;

    /**
     * Creates an untrained transformer block.
     *
     * @param dimension model feature width
     * @param heads number of attention heads
     * @param hiddenDimension feed-forward intermediate width
     * @param seed reproducible initialization seed
     */
    public TransformerBlock(int dimension, int heads, int hiddenDimension, long seed) {
        attentionNormalization = new LayerNormalization(dimension, 1e-5);
        feedForwardNormalization = new LayerNormalization(dimension, 1e-5);
        attention = new MultiHeadAttention(dimension, heads, seed);
        feedForward = new FeedForward(dimension, hiddenDimension, seed + 1);
    }

    /**
     * Computes pre-norm attention and feed-forward residual updates.
     *
     * @param input token vectors [sequence length, dimension]
     * @return contextualized token vectors of the same shape
     */
    public INDArray forward(INDArray input) {
        INDArray afterAttention = input.add(attention.forward(attentionNormalization.forward(input)));
        return afterAttention.add(feedForward.forward(feedForwardNormalization.forward(afterAttention)));
    }
}
