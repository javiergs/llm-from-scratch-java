package com.javiergs.llm.attention;

import java.util.Random;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.NDArrayIndex;

/**
 * Multi-head causal self-attention using ND4J matrix operations.
 * Each head uses a distinct slice of the projected query, key, and value vectors.
 * Parameters are initialized randomly; gradient updates are introduced later.
 *
 * @author Javier Gonzalez-Sanchez
 * @version 0.1.0 (2026-10-31)
 */
public final class MultiHeadAttention {
    private final int dimension;
    private final int heads;
    private final int headDimension;
    private final INDArray queryWeights;
    private final INDArray keyWeights;
    private final INDArray valueWeights;
    private final INDArray outputWeights;

    /**
     * Initializes multi-head attention projections.
     *
     * @param dimension model embedding width, divisible by heads
     * @param heads number of attention heads
     * @param seed initialization seed
     */
    public MultiHeadAttention(int dimension, int heads, long seed) {
        if (dimension <= 0 || heads <= 0 || dimension % heads != 0) {
            throw new IllegalArgumentException("Dimension must be positive and divisible by a positive head count");
        }
        this.dimension = dimension;
        this.heads = heads;
        this.headDimension = dimension / heads;
        Random random = new Random(seed);
        queryWeights = initialize(random);
        keyWeights = initialize(random);
        valueWeights = initialize(random);
        outputWeights = initialize(random);
    }

    private INDArray initialize(Random random) {
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
     * Computes the concatenated head outputs followed by an output projection.
     *
     * @param input matrix of shape [sequence length, dimension]
     * @return contextualized vectors of the same shape
     */
    public INDArray forward(INDArray input) {
        validate(input);
        INDArray queries = input.mmul(queryWeights);
        INDArray keys = input.mmul(keyWeights);
        INDArray values = input.mmul(valueWeights);
        INDArray[] results = new INDArray[heads];
        for (int head = 0; head < heads; head++) {
            INDArray q = sliceHead(queries, head);
            INDArray k = sliceHead(keys, head);
            INDArray v = sliceHead(values, head);
            results[head] = probabilities(q, k).mmul(v);
        }
        return Nd4j.hstack(results).mmul(outputWeights);
    }

    /**
     * Returns the causal probability matrix for a particular head.
     *
     * @param input matrix of shape [sequence length, dimension]
     * @param head zero-based head index
     * @return attention probabilities of shape [sequence length, sequence length]
     */
    public INDArray attentionWeights(INDArray input, int head) {
        validate(input);
        if (head < 0 || head >= heads) {
            throw new IllegalArgumentException("Invalid attention head: " + head);
        }
        return probabilities(sliceHead(input.mmul(queryWeights), head),
                sliceHead(input.mmul(keyWeights), head));
    }

    private INDArray sliceHead(INDArray projected, int head) {
        int start = head * headDimension;
        return projected.get(NDArrayIndex.all(), NDArrayIndex.interval(start, start + headDimension));
    }

    private INDArray probabilities(INDArray queries, INDArray keys) {
        INDArray scores = queries.mmul(keys.transpose()).div(Math.sqrt(headDimension));
        int length = Math.toIntExact(scores.size(0));
        float[][] result = new float[length][length];
        for (int i = 0; i < length; i++) {
            double max = Double.NEGATIVE_INFINITY;
            for (int j = 0; j <= i; j++) {
                max = Math.max(max, scores.getDouble(i, j));
            }
            double sum = 0.0;
            for (int j = 0; j <= i; j++) {
                result[i][j] = (float) Math.exp(scores.getDouble(i, j) - max);
                sum += result[i][j];
            }
            for (int j = 0; j <= i; j++) {
                result[i][j] = (float) (result[i][j] / sum);
            }
        }
        return Nd4j.create(result);
    }

    private void validate(INDArray input) {
        if (input == null) {
            throw new NullPointerException("input");
        }
        if (input.rank() != 2 || input.size(0) == 0 || input.size(1) != dimension) {
            throw new IllegalArgumentException("Expected nonempty rank-2 input with width " + dimension);
        }
    }
}
