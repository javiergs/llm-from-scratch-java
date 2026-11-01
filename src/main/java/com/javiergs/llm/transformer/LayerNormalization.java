package com.javiergs.llm.transformer;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

/**
 * Normalizes each token's feature vector independently.
 * Scale and shift are initialized to one and zero, respectively.
 *
 * @author Javier Gonzalez-Sanchez
 * @version 0.1.0 (2026-10-31)
 */
public final class LayerNormalization {
    private final int dimension;
    private final double epsilon;

    /**
     * Creates a layer normalization operation.
     *
     * @param dimension feature width
     * @param epsilon positive numerical-stability constant
     */
    public LayerNormalization(int dimension, double epsilon) {
        if (dimension <= 0 || !Double.isFinite(epsilon) || epsilon <= 0) {
            throw new IllegalArgumentException("Dimension and epsilon must be positive");
        }
        this.dimension = dimension;
        this.epsilon = epsilon;
    }

    /**
     * Normalizes each row across its features.
     *
     * @param input matrix of shape [sequence length, dimension]
     * @return normalized matrix with the same shape
     */
    public INDArray forward(INDArray input) {
        if (input == null) {
            throw new NullPointerException("input");
        }
        if (input.rank() != 2 || input.size(0) == 0 || input.size(1) != dimension) {
            throw new IllegalArgumentException("Expected nonempty rank-2 input of width " + dimension);
        }
        int length = Math.toIntExact(input.size(0));
        float[][] result = new float[length][dimension];
        for (int row = 0; row < length; row++) {
            double mean = 0;
            for (int col = 0; col < dimension; col++) {
                mean += input.getDouble(row, col);
            }
            mean /= dimension;
            double variance = 0;
            for (int col = 0; col < dimension; col++) {
                double difference = input.getDouble(row, col) - mean;
                variance += difference * difference;
            }
            variance /= dimension;
            double denominator = Math.sqrt(variance + epsilon);
            for (int col = 0; col < dimension; col++) {
                result[row][col] = (float) ((input.getDouble(row, col) - mean) / denominator);
            }
        }
        return Nd4j.create(result);
    }
}
