package com.javiergs.llm.embedding;

import java.util.Objects;
import java.util.Random;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

/**
 * Maps token IDs to trainable embedding vectors stored in an ND4J matrix.
 * This lesson implements the forward lookup only; gradient updates come later.
 *
 * @author Javier Gonzalez-Sanchez
 * @version 0.1.0 (2026-10-31)
 */
public final class TokenEmbedding {
    private final int vocabularySize;
    private final int dimension;
    private final INDArray weights;

    /**
     * Initializes small random embeddings using a reproducible seed.
     *
     * @param vocabularySize number of tokens, must be positive
     * @param dimension vector size, must be positive
     * @param seed random initialization seed
     */
    public TokenEmbedding(int vocabularySize, int dimension, long seed) {
        if (vocabularySize <= 0 || dimension <= 0) {
            throw new IllegalArgumentException("Vocabulary size and dimension must be positive");
        }
        this.vocabularySize = vocabularySize;
        this.dimension = dimension;
        Random random = new Random(seed);
        float[][] values = new float[vocabularySize][dimension];
        for (int i = 0; i < vocabularySize; i++) {
            for (int j = 0; j < dimension; j++) {
                values[i][j] = (float) (random.nextGaussian() * 0.02);
            }
        }
        this.weights = Nd4j.create(values);
    }

    /**
     * Looks up an embedding vector for every input token.
     *
     * @param tokenIds token IDs to embed
     * @return matrix with shape [number of tokens, embedding dimension]
     */
    public INDArray forward(int[] tokenIds) {
        Objects.requireNonNull(tokenIds, "tokenIds");
        float[][] result = new float[tokenIds.length][dimension];
        for (int i = 0; i < tokenIds.length; i++) {
            int tokenId = tokenIds[i];
            if (tokenId < 0 || tokenId >= vocabularySize) {
                throw new IllegalArgumentException("Invalid token ID: " + tokenId);
            }
            for (int j = 0; j < dimension; j++) {
                result[i][j] = weights.getFloat(tokenId, j);
            }
        }
        if (tokenIds.length == 0) {
            return Nd4j.create(org.nd4j.linalg.api.buffer.DataType.FLOAT, 0, dimension);
        }
        return Nd4j.create(result);
    }

    /**
     * Returns a copy of the embedding weights.
     *
     * @return weight matrix of shape [vocabulary size, dimension]
     */
    public INDArray weights() {
        return weights.dup();
    }

    /**
     * Returns the embedding dimension.
     *
     * @return vector dimension
     */
    public int dimension() {
        return dimension;
    }
}
