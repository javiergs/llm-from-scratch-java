package com.javiergs.llm.training;

import org.junit.jupiter.api.Test;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Verifies embedding gradient accumulation and input validation.
 *
 * @author Javier Gonzalez-Sanchez
 * @version 0.1.0 (2026-10-31)
 */
class EmbeddingGradientsTest {
    @Test
    void repeatedTokensAccumulateTheirGradients() {
        INDArray upstream = Nd4j.create(new float[][]{{1, 2}, {3, 4}, {5, 6}});
        INDArray gradient = EmbeddingGradients.tokenWeights(new int[]{1, 0, 1}, upstream, 4);
        assertEquals(6, gradient.getDouble(1, 0), 1e-6);
        assertEquals(8, gradient.getDouble(1, 1), 1e-6);
        assertEquals(3, gradient.getDouble(0, 0), 1e-6);
        assertEquals(0, gradient.getDouble(3, 1), 1e-6);
    }

    @Test
    void positionGradientsMatchRowsAndUnusedRowsAreZero() {
        INDArray upstream = Nd4j.create(new float[][]{{1, 2}, {3, 4}});
        INDArray gradient = EmbeddingGradients.positionWeights(new int[]{2, 2}, upstream, 4);
        assertEquals(1, gradient.getDouble(0, 0), 1e-6);
        assertEquals(4, gradient.getDouble(1, 1), 1e-6);
        assertEquals(0, gradient.getDouble(2, 0), 1e-6);
        assertEquals(0, gradient.getDouble(3, 1), 1e-6);
    }

    @Test
    void tokenGradientMatchesFiniteDifferences() {
        int[] ids = {1, 0, 1};
        INDArray upstream = Nd4j.create(new float[][]{{0.5f, 0.2f}, {-0.4f, 0.1f}, {0.3f, -0.6f}});
        INDArray analytical = EmbeddingGradients.tokenWeights(ids, upstream, 3);
        INDArray weights = Nd4j.create(new float[][]{{0.1f, 0.2f}, {0.3f, 0.4f}, {0.5f, 0.6f}});
        double step = 1e-3;
        for (int token = 0; token < 3; token++) {
            for (int column = 0; column < 2; column++) {
                INDArray plus = weights.dup();
                INDArray minus = weights.dup();
                plus.putScalar(token, column, plus.getDouble(token, column) + step);
                minus.putScalar(token, column, minus.getDouble(token, column) - step);
                double numerical = (lookupDot(plus, ids, upstream)
                        - lookupDot(minus, ids, upstream)) / (2 * step);
                assertEquals(numerical, analytical.getDouble(token, column), 1e-3);
            }
        }
    }

    private static double lookupDot(INDArray weights, int[] ids, INDArray upstream) {
        double result = 0;
        for (int row = 0; row < ids.length; row++) {
            for (int column = 0; column < upstream.size(1); column++) {
                result += weights.getDouble(ids[row], column) * upstream.getDouble(row, column);
            }
        }
        return result;
    }

    @Test
    void rejectsInvalidShapesAndIds() {
        INDArray upstream = Nd4j.ones(2, 2);
        assertThrows(IllegalArgumentException.class,
                () -> EmbeddingGradients.tokenWeights(new int[]{0}, upstream, 3));
        assertThrows(IllegalArgumentException.class,
                () -> EmbeddingGradients.tokenWeights(new int[]{0, 3}, upstream, 3));
        assertThrows(IllegalArgumentException.class,
                () -> EmbeddingGradients.positionWeights(new int[]{0, 1}, upstream, 1));
        assertThrows(NullPointerException.class,
                () -> EmbeddingGradients.tokenWeights(null, upstream, 3));
    }
}
