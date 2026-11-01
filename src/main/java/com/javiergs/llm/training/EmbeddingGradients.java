package com.javiergs.llm.training;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

/**
 * Backpropagates through token lookup and additive positional embeddings.
 * Both tables receive the same upstream gradient at each sequence position.
 *
 * @author Javier Gonzalez-Sanchez
 * @version 0.1.0 (2026-10-31)
 */
public final class EmbeddingGradients {
    private EmbeddingGradients() {
    }

    /**
     * Accumulates gradients into token-table rows, including repeated IDs.
     *
     * @param tokenIds token IDs used by the forward lookup
     * @param upstream gradient of the combined embedding output [sequence, dimension]
     * @param vocabularySize total number of token rows
     * @return token-table gradient [vocabulary size, dimension]
     */
    public static INDArray tokenWeights(int[] tokenIds, INDArray upstream, int vocabularySize) {
        validate(tokenIds, upstream);
        if (vocabularySize <= 0) {
            throw new IllegalArgumentException("Vocabulary size must be positive");
        }
        int dimension = Math.toIntExact(upstream.size(1));
        INDArray gradient = Nd4j.zeros(vocabularySize, dimension);
        for (int position = 0; position < tokenIds.length; position++) {
            int token = tokenIds[position];
            if (token < 0 || token >= vocabularySize) {
                throw new IllegalArgumentException("Invalid token ID: " + token);
            }
            for (int column = 0; column < dimension; column++) {
                gradient.putScalar(token, column,
                        gradient.getDouble(token, column) + upstream.getDouble(position, column));
            }
        }
        return gradient;
    }

    /**
     * Accumulates positional gradients in the corresponding position-table rows.
     * Unused positions receive zero gradient.
     *
     * @param tokenIds input sequence (used to check the row count)
     * @param upstream gradient of the combined embedding output [sequence, dimension]
     * @param contextLength number of available positional rows
     * @return position-table gradient [context length, dimension]
     */
    public static INDArray positionWeights(int[] tokenIds, INDArray upstream, int contextLength) {
        validate(tokenIds, upstream);
        if (contextLength <= 0 || tokenIds.length > contextLength) {
            throw new IllegalArgumentException("Context length must cover the input sequence");
        }
        int dimension = Math.toIntExact(upstream.size(1));
        INDArray gradient = Nd4j.zeros(contextLength, dimension);
        for (int position = 0; position < tokenIds.length; position++) {
            for (int column = 0; column < dimension; column++) {
                gradient.putScalar(position, column, upstream.getDouble(position, column));
            }
        }
        return gradient;
    }

    private static void validate(int[] tokenIds, INDArray upstream) {
        if (tokenIds == null || upstream == null) {
            throw new NullPointerException("tokenIds and upstream must not be null");
        }
        if (upstream.rank() != 2 || upstream.size(0) != tokenIds.length
                || upstream.size(1) <= 0 || tokenIds.length == 0) {
            throw new IllegalArgumentException("Expected a nonempty sequence and matching rank-2 gradient");
        }
        for (long index = 0; index < upstream.length(); index++) {
            if (!Double.isFinite(upstream.getDouble(index))) {
                throw new IllegalArgumentException("Upstream gradients must be finite");
            }
        }
    }
}
