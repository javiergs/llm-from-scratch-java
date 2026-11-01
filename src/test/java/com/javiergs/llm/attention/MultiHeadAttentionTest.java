package com.javiergs.llm.attention;

import org.junit.jupiter.api.Test;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import static org.junit.jupiter.api.Assertions.*;

/**
 * Tests multi-head dimensions, masking, and autoregressive independence.
 *
 * @author Javier Gonzalez-Sanchez
 * @version 0.1.0 (2026-10-31)
 */
class MultiHeadAttentionTest {

    @Test
    void preservesSequenceLengthAndModelWidth() {
        MultiHeadAttention attention = new MultiHeadAttention(4, 2, 42L);
        assertArrayEquals(new long[]{3, 4}, attention.forward(Nd4j.ones(3, 4)).shape());
    }

    @Test
    void eachHeadIsCausalAndNormalized() {
        MultiHeadAttention attention = new MultiHeadAttention(4, 2, 42L);
        INDArray input = Nd4j.create(new float[][]{{1, 0, 0, 1}, {0, 1, 1, 0}, {1, 1, 0, 0}});
        for (int head = 0; head < 2; head++) {
            INDArray weights = attention.attentionWeights(input, head);
            assertArrayEquals(new long[]{3, 3}, weights.shape());
            for (int i = 0; i < 3; i++) {
                double sum = 0;
                for (int j = 0; j < 3; j++) {
                    double weight = weights.getDouble(i, j);
                    if (j > i) {
                        assertEquals(0.0, weight, 0.0);
                    } else {
                        assertTrue(Double.isFinite(weight) && weight >= 0.0);
                    }
                    sum += weight;
                }
                assertEquals(1.0, sum, 1e-6);
            }
        }
    }

    @Test
    void changingFutureTokensCannotChangeEarlierOutputs() {
        MultiHeadAttention attention = new MultiHeadAttention(4, 2, 7L);
        INDArray a = Nd4j.create(new float[][]{{1, 2, 3, 4}, {2, 3, 4, 5}, {3, 4, 5, 6}});
        INDArray b = Nd4j.create(new float[][]{{1, 2, 3, 4}, {2, 3, 4, 5}, {30, 40, 50, 60}});
        INDArray outputA = attention.forward(a);
        INDArray outputB = attention.forward(b);
        for (int i = 0; i < 2; i++) {
            for (int j = 0; j < 4; j++) {
                assertEquals(outputA.getDouble(i, j), outputB.getDouble(i, j), 1e-6);
            }
        }
    }

    @Test
    void seededInitializationIsReproducible() {
        INDArray input = Nd4j.ones(3, 4);
        assertEquals(new MultiHeadAttention(4, 2, 9L).forward(input),
                new MultiHeadAttention(4, 2, 9L).forward(input));
    }

    @Test
    void validatesConfigurationAndInputs() {
        assertThrows(IllegalArgumentException.class, () -> new MultiHeadAttention(3, 2, 1L));
        assertThrows(IllegalArgumentException.class, () -> new MultiHeadAttention(4, 0, 1L));
        MultiHeadAttention attention = new MultiHeadAttention(4, 2, 1L);
        assertThrows(IllegalArgumentException.class, () -> attention.forward(Nd4j.ones(2, 3)));
        assertThrows(IllegalArgumentException.class, () -> attention.attentionWeights(Nd4j.ones(2, 4), 2));
        assertThrows(NullPointerException.class, () -> attention.forward(null));
    }
}
