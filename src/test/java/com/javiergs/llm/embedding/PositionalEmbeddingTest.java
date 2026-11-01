package com.javiergs.llm.embedding;

import org.junit.jupiter.api.Test;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Verifies absolute positional embedding lookup and composition.
 *
 * @author Javier Gonzalez-Sanchez
 * @version 0.1.0 (2026-10-31)
 */
class PositionalEmbeddingTest {

    @Test
    void returnsExpectedShape() {
        PositionalEmbedding positions = new PositionalEmbedding(8, 4, 1L);
        assertArrayEquals(new long[]{3, 4}, positions.forward(3).shape());
        assertArrayEquals(new long[]{8, 4}, positions.weights().shape());
    }

    @Test
    void initializationIsReproducible() {
        assertEquals(new PositionalEmbedding(4, 3, 9L).weights(),
                new PositionalEmbedding(4, 3, 9L).weights());
    }

    @Test
    void combinesVectorsWithoutChangingInputs() {
        PositionalEmbedding positions = new PositionalEmbedding(5, 2, 5L);
        INDArray tokens = Nd4j.ones(3, 2);
        INDArray before = tokens.dup();
        INDArray combined = positions.addTo(tokens);
        INDArray expected = before.add(positions.forward(3));
        assertEquals(expected, combined);
        assertEquals(before, tokens);
    }

    @Test
    void sameTokenAtDifferentPositionsGetsDifferentCombinedVectors() {
        TokenEmbedding tokens = new TokenEmbedding(3, 4, 7L);
        PositionalEmbedding positions = new PositionalEmbedding(3, 4, 8L);
        INDArray combined = positions.addTo(tokens.forward(new int[]{1, 1}));
        assertNotEquals(combined.getRow(0).dup(), combined.getRow(1).dup());
    }

    @Test
    void rejectsInvalidSizesAndShapes() {
        assertThrows(IllegalArgumentException.class, () -> new PositionalEmbedding(0, 2, 1L));
        assertThrows(IllegalArgumentException.class, () -> new PositionalEmbedding(2, 0, 1L));
        PositionalEmbedding positions = new PositionalEmbedding(2, 3, 1L);
        assertThrows(IllegalArgumentException.class, () -> positions.forward(-1));
        assertThrows(IllegalArgumentException.class, () -> positions.forward(3));
        assertThrows(IllegalArgumentException.class, () -> positions.addTo(Nd4j.ones(2, 2)));
        assertThrows(IllegalArgumentException.class, () -> positions.addTo(Nd4j.ones(3, 3)));
    }

    @Test
    void protectsInternalWeights() {
        PositionalEmbedding positions = new PositionalEmbedding(3, 2, 1L);
        INDArray before = positions.weights();
        positions.weights().assign(0);
        assertEquals(before, positions.weights());
    }
}
