package com.javiergs.llm.training;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

/**
 * Backpropagates through single-head causal scaled dot-product attention.
 * The calculation matches the masking and scaling in CausalSelfAttention.
 *
 * @author Javier Gonzalez-Sanchez
 * @version 0.1.0 (2026-10-31)
 */
public final class CausalAttentionGradients {
    private CausalAttentionGradients() {
    }

    /**
     * Holds gradients for the input and the three projection matrices.
     *
     * @param inputGradient derivative with respect to input
     * @param queryWeightsGradient derivative with respect to query weights
     * @param keyWeightsGradient derivative with respect to key weights
     * @param valueWeightsGradient derivative with respect to value weights
     */
    public record Gradients(INDArray inputGradient, INDArray queryWeightsGradient,
                            INDArray keyWeightsGradient, INDArray valueWeightsGradient) {
    }

    /**
     * Differentiates causal attention output = softmax(Q K^T / sqrt(D)) V.
     * Masked future positions contribute exactly zero gradient.
     *
     * @param input token vectors [sequence, dimension]
     * @param queryWeights query projection [dimension, dimension]
     * @param keyWeights key projection [dimension, dimension]
     * @param valueWeights value projection [dimension, dimension]
     * @param upstream output gradient [sequence, dimension]
     * @return input and projection weight gradients
     */
    public static Gradients backward(INDArray input, INDArray queryWeights,
                                     INDArray keyWeights, INDArray valueWeights,
                                     INDArray upstream) {
        if (input == null || queryWeights == null || keyWeights == null
                || valueWeights == null || upstream == null) {
            throw new NullPointerException("Inputs and gradients must not be null");
        }
        if (input.rank() != 2 || upstream.rank() != 2 || input.size(0) == 0
                || input.size(1) == 0 || upstream.size(0) != input.size(0)
                || upstream.size(1) != input.size(1)) {
            throw new IllegalArgumentException("Expected matching nonempty input and upstream matrices");
        }
        int dimension = Math.toIntExact(input.size(1));
        for (INDArray weights : new INDArray[]{queryWeights, keyWeights, valueWeights}) {
            if (weights.rank() != 2 || weights.size(0) != dimension || weights.size(1) != dimension) {
                throw new IllegalArgumentException("Projection matrices must be square with input width");
            }
        }
        int length = Math.toIntExact(input.size(0));
        INDArray q = input.mmul(queryWeights);
        INDArray k = input.mmul(keyWeights);
        INDArray v = input.mmul(valueWeights);
        INDArray scores = q.mmul(k.transpose()).div(Math.sqrt(dimension));
        INDArray probabilities = Nd4j.zeros(length, length);
        for (int row = 0; row < length; row++) {
            double maximum = Double.NEGATIVE_INFINITY;
            for (int col = 0; col <= row; col++) {
                maximum = Math.max(maximum, scores.getDouble(row, col));
            }
            double denominator = 0;
            for (int col = 0; col <= row; col++) {
                denominator += Math.exp(scores.getDouble(row, col) - maximum);
            }
            for (int col = 0; col <= row; col++) {
                probabilities.putScalar(row, col,
                        Math.exp(scores.getDouble(row, col) - maximum) / denominator);
            }
        }
        INDArray dv = probabilities.transpose().mmul(upstream);
        INDArray dp = upstream.mmul(v.transpose());
        INDArray ds = Nd4j.zeros(length, length);
        for (int row = 0; row < length; row++) {
            double weightedSum = 0;
            for (int col = 0; col <= row; col++) {
                weightedSum += probabilities.getDouble(row, col) * dp.getDouble(row, col);
            }
            for (int col = 0; col <= row; col++) {
                ds.putScalar(row, col, probabilities.getDouble(row, col)
                        * (dp.getDouble(row, col) - weightedSum));
            }
        }
        INDArray dq = ds.mmul(k).div(Math.sqrt(dimension));
        INDArray dk = ds.transpose().mmul(q).div(Math.sqrt(dimension));
        INDArray dWq = input.transpose().mmul(dq);
        INDArray dWk = input.transpose().mmul(dk);
        INDArray dWv = input.transpose().mmul(dv);
        INDArray dx = dq.mmul(queryWeights.transpose())
                .add(dk.mmul(keyWeights.transpose()))
                .add(dv.mmul(valueWeights.transpose()));
        return new Gradients(dx, dWq, dWk, dWv);
    }
}
