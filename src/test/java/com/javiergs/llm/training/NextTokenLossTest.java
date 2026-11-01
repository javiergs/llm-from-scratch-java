package com.javiergs.llm.training;

import org.junit.jupiter.api.Test;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import static org.junit.jupiter.api.Assertions.*;

/**
 * Tests next-token target alignment and stable cross-entropy.
 *
 * @author Javier Gonzalez-Sanchez
 * @version 0.1.0 (2026-10-31)
 */
class NextTokenLossTest {

    @Test
    void uniformScoresHaveLogVocabularySizeLoss() {
        INDArray logits = Nd4j.create(new float[][]{{0, 0, 0}, {0, 0, 0}, {0, 0, 0}});
        assertEquals(Math.log(3), NextTokenLoss.mean(logits, new int[]{0, 1, 2}), 1e-6);
    }

    @Test
    void correctNextTokenScoresReduceLoss() {
        INDArray good = Nd4j.create(new float[][]{{0, 8, 0}, {0, 0, 8}, {0, 0, 0}});
        INDArray bad = Nd4j.create(new float[][]{{8, 0, 0}, {8, 0, 0}, {0, 0, 0}});
        int[] ids = {0, 1, 2};
        assertTrue(NextTokenLoss.mean(good, ids) < NextTokenLoss.mean(bad, ids));
    }

    @Test
    void ignoresFinalLogitRow() {
        INDArray a = Nd4j.create(new float[][]{{0, 2}, {2, 0}, {0, 0}});
        INDArray b = Nd4j.create(new float[][]{{0, 2}, {2, 0}, {1000, -1000}});
        int[] ids = {0, 1, 0};
        assertEquals(NextTokenLoss.mean(a, ids), NextTokenLoss.mean(b, ids), 1e-10);
    }

    @Test
    void largeFiniteScoresRemainNumericallyStable() {
        INDArray logits = Nd4j.create(new float[][]{{10000, 10001}, {0, 0}});
        double loss = NextTokenLoss.mean(logits, new int[]{0, 1});
        assertTrue(Double.isFinite(loss));
        assertEquals(Math.log1p(Math.exp(-1)), loss, 1e-5);
    }

    @Test
    void rejectsInvalidInputs() {
        INDArray logits = Nd4j.create(new float[][]{{1, 2}, {3, 4}});
        assertThrows(IllegalArgumentException.class, () -> NextTokenLoss.mean(logits, new int[]{0}));
        assertThrows(IllegalArgumentException.class, () -> NextTokenLoss.mean(logits, new int[]{0, 2}));
        assertThrows(IllegalArgumentException.class, () -> NextTokenLoss.mean(logits, new int[]{0, -1}));
        assertThrows(NullPointerException.class, () -> NextTokenLoss.mean(null, new int[]{0, 1}));
        assertThrows(NullPointerException.class, () -> NextTokenLoss.mean(logits, null));
    }
}
