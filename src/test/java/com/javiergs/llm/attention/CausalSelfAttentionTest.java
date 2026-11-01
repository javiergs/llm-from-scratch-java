package com.javiergs.llm.attention;

import org.junit.jupiter.api.Test;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import static org.junit.jupiter.api.Assertions.*;

/**
 * Checks causal masking, normalization, reproducibility, and input validation.
 *
 * @author Javier Gonzalez-Sanchez
 * @version 0.1.0 (2026-10-31)
 */
class CausalSelfAttentionTest {

    @Test
    void outputHasSameShapeAsInput() {
        INDArray input = Nd4j.ones(3, 4);
        assertArrayEquals(new long[]{3, 4}, new CausalSelfAttention(4, 7L).forward(input).shape());
    }

    @Test
    void weightsAreCausalAndNormalized() {
        INDArray input = Nd4j.ones(4, 3);
        INDArray weights = new CausalSelfAttention(3, 7L).attentionWeights(input);
        for (int i = 0; i < 4; i++) {
            double sum = 0;
            for (int j = 0; j < 4; j++) {
                double value = weights.getDouble(i, j);
                if (j > i) {
                    assertEquals(0.0, value, 0.0);
                } else {
                    assertTrue(value >= 0 && Double.isFinite(value));
                }
                sum += value;
            }
            assertEquals(1.0, sum, 1e-6);
        }
    }

    @Test
    void futureTokensDoNotAffectEarlierOutputs() {
        CausalSelfAttention attention = new CausalSelfAttention(2, 13L);
        INDArray first = Nd4j.create(new float[][]{{1, 2}, {3, 4}, {5, 6}});
        INDArray second = Nd4j.create(new float[][]{{1, 2}, {3, 4}, {50, 60}});
        INDArray a = attention.forward(first);
        INDArray b = attention.forward(second);
        for (int i = 0; i < 2; i++) {
            for (int j = 0; j < 2; j++) {
                assertEquals(a.getDouble(i, j), b.getDouble(i, j), 1e-6);
            }
        }
    }

    @Test
    void sameSeedProducesSameOutputs() {
        INDArray input = Nd4j.ones(3, 2);
        assertEquals(new CausalSelfAttention(2, 9L).forward(input),
                new CausalSelfAttention(2, 9L).forward(input));
    }

    @Test
    void rejectsInvalidInputs() {
        assertThrows(IllegalArgumentException.class, () -> new CausalSelfAttention(0, 1L));
        CausalSelfAttention attention = new CausalSelfAttention(3, 1L);
        assertThrows(IllegalArgumentException.class, () -> attention.forward(Nd4j.ones(2, 2)));
        assertThrows(IllegalArgumentException.class, () -> attention.forward(Nd4j.ones(3)));
        assertThrows(NullPointerException.class, () -> attention.forward(null));
    }
}
