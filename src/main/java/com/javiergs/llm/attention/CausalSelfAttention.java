package com.javiergs.llm.attention;

import java.util.Random;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

/**
 * Implements single-head causal scaled dot-product self-attention.
 * The Q, K, and V projection matrices are initialized but not trained.
 *
 * @author Javier Gonzalez-Sanchez
 * @version 0.1.0 (2026-10-31)
 */
public final class CausalSelfAttention {
    private final int dimension;
    private final INDArray queryWeights;
    private final INDArray keyWeights;
    private final INDArray valueWeights;

    /**
     * Initializes independent query, key, and value projection matrices.
     *
     * @param dimension input and output embedding width
     * @param seed random initialization seed
     */
    public CausalSelfAttention(int dimension, long seed) {
        if (dimension <= 0) {
            throw new IllegalArgumentException("Dimension must be positive");
        }
        this.dimension = dimension;
        Random random = new Random(seed);
        queryWeights = initialize(dimension, random);
        keyWeights = initialize(dimension, random);
        valueWeights = initialize(dimension, random);
    }

    private static INDArray initialize(int dimension, Random random) {
        float[][] values = new float[dimension][dimension];
        double scale = 1.0 / Math.sqrt(dimension);
        for (int i = 0; i < dimension; i++) {
            for (int j = 0; j < dimension; j++) {
                values[i][j] = (float) (random.nextGaussian() * scale);
            }
        }
        return Nd4j.create(values);
    }

    /**
     * Computes causal attention outputs for a sequence.
     *
     * @param input matrix with shape [sequence length, dimension]
     * @return contextualized vectors of the same shape
     */
    public INDArray forward(INDArray input) {
        return attention(input, false);
    }

    /**
     * Returns causal attention probabilities for educational inspection.
     * Each row sums to one; entries above the diagonal are exactly zero.
     *
     * @param input matrix with shape [sequence length, dimension]
     * @return probability matrix [sequence length, sequence length]
     */
    public INDArray attentionWeights(INDArray input) {
        return attention(input, true);
    }

    private INDArray attention(INDArray input, boolean returnWeights) {
        if (input == null) {
            throw new NullPointerException("input");
        }
        if (input.rank() != 2 || input.size(1) != dimension || input.size(0) == 0) {
            throw new IllegalArgumentException("Expected nonempty rank-2 matrix of width " + dimension);
        }
        int length = Math.toIntExact(input.size(0));
        INDArray queries = input.mmul(queryWeights);
        INDArray keys = input.mmul(keyWeights);
        INDArray values = input.mmul(valueWeights);
        INDArray scores = queries.mmul(keys.transpose()).div(Math.sqrt(dimension));

        // Stable, causal softmax: normalize only over positions j <= i.
        float[][] probabilities = new float[length][length];
        for (int i = 0; i < length; i++) {
            double max = Double.NEGATIVE_INFINITY;
            for (int j = 0; j <= i; j++) {
                max = Math.max(max, scores.getDouble(i, j));
            }
            double sum = 0.0;
            for (int j = 0; j <= i; j++) {
                double weight = Math.exp(scores.getDouble(i, j) - max);
                probabilities[i][j] = (float) weight;
                sum += weight;
            }
            for (int j = 0; j <= i; j++) {
                probabilities[i][j] /= (float) sum;
            }
        }
        INDArray weights = Nd4j.create(probabilities);
        return returnWeights ? weights : weights.mmul(values);
    }
}
