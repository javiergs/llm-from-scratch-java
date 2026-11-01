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
    /**
     * Returns copies of six trainable projection matrices in Q, K, V,
     * attention output, feed-forward input, feed-forward output order.
     *
     * @return six independent matrix copies
     */
    public INDArray[] parameters() {
        INDArray[] attentionParameters = attention.parameters();
        INDArray[] feedForwardParameters = feedForward.parameters();
        return new INDArray[]{attentionParameters[0], attentionParameters[1],
                attentionParameters[2], attentionParameters[3],
                feedForwardParameters[0], feedForwardParameters[1]};
    }

    /**
     * Replaces all six projections using shape-validated copies.
     *
     * @param parameters six matrices in the order returned by parameters()
     */
    public void setParameters(INDArray[] parameters) {
        if (parameters == null || parameters.length != 6) {
            throw new IllegalArgumentException("Expected six transformer matrices");
        }
        INDArray[] current = parameters();
        for (int i = 0; i < 6; i++) {
            if (parameters[i] == null || !java.util.Arrays.equals(
                    parameters[i].shape(), current[i].shape())) {
                throw new IllegalArgumentException("Invalid transformer matrix at index " + i);
            }
        }
        attention.setParameters(java.util.Arrays.copyOfRange(parameters, 0, 4));
        feedForward.setParameters(java.util.Arrays.copyOfRange(parameters, 4, 6));
    }

}
