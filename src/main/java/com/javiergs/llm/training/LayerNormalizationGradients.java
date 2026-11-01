package com.javiergs.llm.training;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

/**
 * Backpropagates through row-wise layer normalization without affine parameters.
 * Each row is normalized using population variance, matching LayerNormalization.
 *
 * @author Javier Gonzalez-Sanchez
 * @version 0.1.0 (2026-10-31)
 */
public final class LayerNormalizationGradients {
    private LayerNormalizationGradients() {
    }

    /**
     * Computes the gradient with respect to the input of layer normalization.
     * The derivative accounts for changes to both the row mean and variance.
     *
     * @param input original pre-normalization vectors [sequence, dimension]
     * @param upstream gradient with respect to normalized output, same shape
     * @param epsilon positive stability constant used by the forward pass
     * @return gradient with respect to input, same shape
     */
    public static INDArray inputGradient(INDArray input, INDArray upstream, double epsilon) {
        if (input == null || upstream == null) {
            throw new NullPointerException("input and upstream must not be null");
        }
        if (input.rank() != 2 || upstream.rank() != 2 || input.size(0) == 0
                || input.size(1) == 0 || !java.util.Arrays.equals(input.shape(), upstream.shape())
                || !Double.isFinite(epsilon) || epsilon <= 0) {
            throw new IllegalArgumentException("Expected matching nonempty matrices and positive epsilon");
        }
        int rows = Math.toIntExact(input.size(0));
        int width = Math.toIntExact(input.size(1));
        INDArray gradient = Nd4j.zeros(rows, width);
        for (int row = 0; row < rows; row++) {
            double mean = 0;
            for (int col = 0; col < width; col++) {
                double value = input.getDouble(row, col);
                if (!Double.isFinite(value) || !Double.isFinite(upstream.getDouble(row, col))) {
                    throw new IllegalArgumentException("Inputs and gradients must be finite");
                }
                mean += value;
            }
            mean /= width;
            double variance = 0;
            for (int col = 0; col < width; col++) {
                double centered = input.getDouble(row, col) - mean;
                variance += centered * centered;
            }
            variance /= width;
            double inverseStd = 1.0 / Math.sqrt(variance + epsilon);
            double upstreamMean = 0;
            double upstreamTimesNormalizedMean = 0;
            for (int col = 0; col < width; col++) {
                double normalized = (input.getDouble(row, col) - mean) * inverseStd;
                double derivative = upstream.getDouble(row, col);
                upstreamMean += derivative;
                upstreamTimesNormalizedMean += derivative * normalized;
            }
            upstreamMean /= width;
            upstreamTimesNormalizedMean /= width;
            for (int col = 0; col < width; col++) {
                double normalized = (input.getDouble(row, col) - mean) * inverseStd;
                double derivative = (upstream.getDouble(row, col) - upstreamMean
                        - normalized * upstreamTimesNormalizedMean) * inverseStd;
                gradient.putScalar(row, col, derivative);
            }
        }
        return gradient;
    }
}
