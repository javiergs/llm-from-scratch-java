package com.javiergs.llm.embedding;

import org.junit.jupiter.api.Test;
import org.nd4j.linalg.api.ndarray.INDArray;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Verifies token embedding dimensions, lookup behavior, and reproducibility.
 *
 * @author Javier Gonzalez-Sanchez
 * @version 0.1.0 (2026-10-31)
 */
class TokenEmbeddingTest {

    @Test
    void returnsOneVectorPerToken() {
        TokenEmbedding embedding = new TokenEmbedding(5, 3, 42L);
        INDArray result = embedding.forward(new int[]{1, 2, 1});
        assertArrayEquals(new long[]{3, 3}, result.shape());
        for (int j = 0; j < 3; j++) {
            assertEquals(result.getDouble(0, j), result.getDouble(2, j), 1e-7);
        }
    }

    @Test
    void lookupMatchesWeightMatrix() {
        TokenEmbedding embedding = new TokenEmbedding(4, 2, 7L);
        INDArray weights = embedding.weights();
        INDArray result = embedding.forward(new int[]{3, 0});
        for (int j = 0; j < 2; j++) {
            assertEquals(weights.getDouble(3, j), result.getDouble(0, j), 1e-7);
            assertEquals(weights.getDouble(0, j), result.getDouble(1, j), 1e-7);
        }
    }

    @Test
    void initializationIsReproducible() {
        TokenEmbedding first = new TokenEmbedding(4, 3, 99L);
        TokenEmbedding second = new TokenEmbedding(4, 3, 99L);
        assertEquals(first.weights(), second.weights());
    }

    @Test
    void rejectsInvalidDimensions() {
        assertThrows(IllegalArgumentException.class, () -> new TokenEmbedding(0, 3, 1L));
        assertThrows(IllegalArgumentException.class, () -> new TokenEmbedding(3, 0, 1L));
    }

    @Test
    void rejectsUnknownTokenIds() {
        TokenEmbedding embedding = new TokenEmbedding(3, 2, 1L);
        assertThrows(IllegalArgumentException.class, () -> embedding.forward(new int[]{3}));
        assertThrows(IllegalArgumentException.class, () -> embedding.forward(new int[]{-1}));
    }

    @Test
    void protectsInternalWeights() {
        TokenEmbedding embedding = new TokenEmbedding(3, 2, 1L);
        INDArray original = embedding.weights();
        INDArray copy = embedding.weights();
        copy.assign(0);
        assertEquals(original, embedding.weights());
    }
}
