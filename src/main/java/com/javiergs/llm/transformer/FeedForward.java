package com.javiergs.llm.transformer;

import java.util.Random;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

/**
 * Position-wise two-layer feed-forward network with GELU activation.
 * Each token is processed independently, preserving causal behavior.
 *
 * @author Javier Gonzalez-Sanchez
 * @version 0.1.0 (2026-10-31)
 */
public final class FeedForward {
    private final int dimension;
    private final int hiddenDimension;
    private final INDArray inputWeights;
    private final INDArray outputWeights;

    /**
     * Initializes the feed-forward projection matrices.
     *
     * @param dimension input and output width
     * @param hiddenDimension intermediate feature width
     * @param seed initialization seed
     */
    public FeedForward(int dimension, int hiddenDimension, long seed) {
        if (dimension <= 0 || hiddenDimension <= 0) {
            throw new IllegalArgumentException("Dimensions must be positive");
        }
        this.dimension = dimension;
        this.hiddenDimension = hiddenDimension;
        Random random = new Random(seed);
        inputWeights = initialize(dimension, hiddenDimension, random);
        outputWeights = initialize(hiddenDimension, dimension, random);
    }

    private static INDArray initialize(int rows, int columns, Random random) {
        float[][] values = new float[rows][columns];
        double scale = 1.0 / Math.sqrt(rows);
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < columns; j++) {
                values[i][j] = (float) (random.nextGaussian() * scale);
            }
        }
        return Nd4j.create(values);
    }

    /**
     * Applies the first projection, GELU, and second projection.
     *
     * @param input token vectors of shape [sequence length, dimension]
     * @return transformed token vectors of the same shape
     */
    public INDArray forward(INDArray input) {
        if (input == null) {
            throw new NullPointerException("input");
        }
        if (input.rank() != 2 || input.size(0) == 0 || input.size(1) != dimension) {
            throw new IllegalArgumentException("Expected nonempty rank-2 input of width " + dimension);
        }
        INDArray hidden = input.mmul(inputWeights);
        int length = Math.toIntExact(hidden.size(0));
        float[][] activated = new float[length][hiddenDimension];
        for (int row = 0; row < length; row++) {
            for (int col = 0; col < hiddenDimension; col++) {
                double x = hidden.getDouble(row, col);
                // Tanh approximation used by many educational GPT implementations.
                activated[row][col] = (float) (0.5 * x * (1.0
                        + Math.tanh(Math.sqrt(2.0 / Math.PI) * (x + 0.044715 * x * x * x))));
            }
        }
        return Nd4j.create(activated).mmul(outputWeights);
    }
}
