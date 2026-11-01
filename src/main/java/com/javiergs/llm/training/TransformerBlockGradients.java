package com.javiergs.llm.training;

import org.nd4j.linalg.api.ndarray.INDArray;

/**
 * Composes local backward rules for a pre-normalization transformer block.
 * Attention and feed-forward weight matrices are passed explicitly.
 *
 * @author Javier Gonzalez-Sanchez
 * @version 0.1.0 (2026-10-31)
 */
public final class TransformerBlockGradients {
    private TransformerBlockGradients() {
    }

    /**
     * Groups the block's parameter matrices without exposing model internals.
     *
     * @param query query projection
     * @param key key projection
     * @param value value projection
     * @param attentionOutput attention output projection
     * @param feedForwardInput first feed-forward projection
     * @param feedForwardOutput second feed-forward projection
     */
    public record Weights(INDArray query, INDArray key, INDArray value,
                          INDArray attentionOutput, INDArray feedForwardInput,
                          INDArray feedForwardOutput) {
    }

    /**
     * Holds gradients for the block input and both sublayers.
     *
     * @param inputGradient derivative with respect to block input
     * @param attention derivatives of the attention sublayer
     * @param feedForward derivatives of the feed-forward sublayer
     */
    public record Gradients(INDArray inputGradient,
                            MultiHeadAttentionGradients.Gradients attention,
                            FeedForwardGradients.Gradients feedForward) {
    }

    /**
     * Applies the chain rule to both residual paths in a pre-norm block.
     *
     * @param input block input [sequence, dimension]
     * @param weights projection weights for both sublayers
     * @param heads number of attention heads
     * @param epsilon normalization stability constant
     * @param upstream derivative with respect to block output
     * @return block input and parameter gradients
     */
    public static Gradients backward(INDArray input, Weights weights, int heads,
                                     double epsilon, INDArray upstream) {
        if (input == null || weights == null || upstream == null) {
            throw new NullPointerException("Input, weights and upstream must not be null");
        }
        if (input.rank() != 2 || input.size(0) == 0 || input.size(1) == 0
                || upstream.rank() != 2 || upstream.size(0) != input.size(0)
                || upstream.size(1) != input.size(1)) {
            throw new IllegalArgumentException("Input and upstream must have matching nonempty shapes");
        }
        if (heads <= 0 || input.size(1) % heads != 0
                || !Double.isFinite(epsilon) || epsilon <= 0) {
            throw new IllegalArgumentException("Invalid head count or epsilon");
        }
        int dimension = Math.toIntExact(input.size(1));
        INDArray norm1 = normalize(input, epsilon);
        INDArray attentionOutput = attentionForward(norm1, weights, heads);
        INDArray afterAttention = input.add(attentionOutput);
        INDArray norm2 = normalize(afterAttention, epsilon);

        FeedForwardGradients.Gradients ff = FeedForwardGradients.backward(
                norm2, weights.feedForwardInput(), weights.feedForwardOutput(), upstream);
        // Second residual: d(afterAttention) = upstream + d(norm2)/d(afterAttention).
        INDArray dAfterAttention = upstream.add(
                LayerNormalizationGradients.inputGradient(afterAttention, ff.inputGradient(), epsilon));
        MultiHeadAttentionGradients.Gradients attention = MultiHeadAttentionGradients.backward(
                norm1, weights.query(), weights.key(), weights.value(),
                weights.attentionOutput(), heads, dAfterAttention);
        // First residual: d(input) = d(afterAttention) + d(norm1)/d(input).
        INDArray dInput = dAfterAttention.add(
                LayerNormalizationGradients.inputGradient(input, attention.inputGradient(), epsilon));
        return new Gradients(dInput, attention, ff);
    }

    private static INDArray normalize(INDArray input, double epsilon) {
        return new com.javiergs.llm.transformer.LayerNormalization(
                Math.toIntExact(input.size(1)), epsilon).forward(input);
    }

    private static INDArray attentionForward(INDArray input, Weights w, int heads) {
        int dimension = Math.toIntExact(input.size(1));
        int headWidth = dimension / heads;
        int length = Math.toIntExact(input.size(0));
        INDArray q = input.mmul(w.query());
        INDArray k = input.mmul(w.key());
        INDArray v = input.mmul(w.value());
        INDArray concatenated = org.nd4j.linalg.factory.Nd4j.zeros(length, dimension);
        for (int head = 0; head < heads; head++) {
            int start = head * headWidth;
            org.nd4j.linalg.indexing.INDArrayIndex rows = org.nd4j.linalg.indexing.NDArrayIndex.all();
            org.nd4j.linalg.indexing.INDArrayIndex cols =
                    org.nd4j.linalg.indexing.NDArrayIndex.interval(start, start + headWidth);
            INDArray qh = q.get(rows, cols);
            INDArray kh = k.get(org.nd4j.linalg.indexing.NDArrayIndex.all(),
                    org.nd4j.linalg.indexing.NDArrayIndex.interval(start, start + headWidth));
            INDArray vh = v.get(org.nd4j.linalg.indexing.NDArrayIndex.all(),
                    org.nd4j.linalg.indexing.NDArrayIndex.interval(start, start + headWidth));
            INDArray scores = qh.mmul(kh.transpose()).div(Math.sqrt(headWidth));
            INDArray p = org.nd4j.linalg.factory.Nd4j.zeros(length, length);
            for (int i = 0; i < length; i++) {
                double maximum = Double.NEGATIVE_INFINITY;
                for (int j = 0; j <= i; j++) {
                    maximum = Math.max(maximum, scores.getDouble(i, j));
                }
                double sum = 0;
                for (int j = 0; j <= i; j++) {
                    sum += Math.exp(scores.getDouble(i, j) - maximum);
                }
                for (int j = 0; j <= i; j++) {
                    p.putScalar(i, j, Math.exp(scores.getDouble(i, j) - maximum) / sum);
                }
            }
            concatenated.get(org.nd4j.linalg.indexing.NDArrayIndex.all(),
                    org.nd4j.linalg.indexing.NDArrayIndex.interval(start, start + headWidth))
                    .assign(p.mmul(vh));
        }
        return concatenated.mmul(w.attentionOutput());
    }
}
