package com.javiergs.llm.embedding;

import java.util.Random;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

/**
 * Stores learned-style, randomly initialized absolute positional embeddings.
 * Values are fixed until a future training lesson implements updates.
 *
 * @author Javier Gonzalez-Sanchez
 * @version 0.1.0 (2026-10-31)
 */
public final class PositionalEmbedding {
    private final int contextLength;
    private final int dimension;
    private final INDArray weights;

    /**
     * Initializes one embedding vector per supported position.
     *
     * @param contextLength maximum number of positions
     * @param dimension embedding width
     * @param seed reproducible initialization seed
     */
    public PositionalEmbedding(int contextLength, int dimension, long seed) {
        if (contextLength <= 0 || dimension <= 0) {
            throw new IllegalArgumentException("Context length and dimension must be positive");
        }
        this.contextLength = contextLength;
        this.dimension = dimension;
        Random random = new Random(seed);
        float[][] values = new float[contextLength][dimension];
        for (int position = 0; position < contextLength; position++) {
            for (int column = 0; column < dimension; column++) {
                values[position][column] = (float) (random.nextGaussian() * 0.02);
            }
        }
        this.weights = Nd4j.create(values);
    }

    /**
     * Returns positional vectors for positions zero through length minus one.
     *
     * @param length number of positions to retrieve
     * @return matrix with shape [length, dimension]
     */
    public INDArray forward(int length) {
        if (length < 0 || length > contextLength) {
            throw new IllegalArgumentException("Sequence length must be between 0 and " + contextLength);
        }
        if (length == 0) {
            return Nd4j.create(org.nd4j.linalg.api.buffer.DataType.FLOAT, 0, dimension);
        }
        float[][] values = new float[length][dimension];
        for (int i = 0; i < length; i++) {
            for (int j = 0; j < dimension; j++) {
                values[i][j] = weights.getFloat(i, j);
            }
        }
        return Nd4j.create(values);
    }

    /**
     * Adds position vectors to token vectors without modifying the input.
     *
     * @param tokenVectors matrix of shape [sequence length, dimension]
     * @return combined token and position vectors
     */
    public INDArray addTo(INDArray tokenVectors) {
        if (tokenVectors == null) {
            throw new NullPointerException("tokenVectors");
        }
        if (tokenVectors.rank() != 2 || tokenVectors.size(1) != dimension) {
            throw new IllegalArgumentException("Expected a rank-2 token matrix of width " + dimension);
        }
        if (tokenVectors.size(0) > contextLength) {
            throw new IllegalArgumentException("Sequence exceeds maximum context length");
        }
        return tokenVectors.add(forward(Math.toIntExact(tokenVectors.size(0))));
    }

    /**
     * Returns a defensive copy of the position embedding weights.
     *
     * @return matrix with shape [context length, dimension]
     */
    public INDArray weights() {
        return weights.dup();
    }
    /**
     * Replaces the positional embedding table after checking its shape.
     *
     * @param replacement new positional weights
     */
    public void setWeights(INDArray replacement) {
        if (replacement == null || !java.util.Arrays.equals(replacement.shape(), weights.shape())) {
            throw new IllegalArgumentException("Invalid positional embedding shape");
        }
        weights.assign(replacement);
    }

}
