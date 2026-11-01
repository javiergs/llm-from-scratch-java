package com.javiergs.llm.training;

import com.javiergs.llm.transformer.LayerNormalization;
import org.junit.jupiter.api.Test;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import static org.junit.jupiter.api.Assertions.*;

/**
 * Tests layer-normalization backward math against the existing forward pass.
 *
 * @author Javier Gonzalez-Sanchez
 * @version 0.1.0 (2026-10-31)
 */
class LayerNormalizationGradientsTest {
    private static final double EPSILON = 1e-5;

    @Test
    void matchesFiniteDifferencesForMultipleRows() {
        INDArray input = Nd4j.create(new float[][]{
                {0.2f, -0.3f, 0.7f, 1.1f},
                {-0.8f, 0.4f, 0.1f, 0.9f}
        });
        INDArray upstream = Nd4j.create(new float[][]{
                {0.5f, -0.1f, 0.2f, -0.4f},
                {-0.3f, 0.8f, 0.1f, 0.2f}
        });
        INDArray analytical = LayerNormalizationGradients.inputGradient(input, upstream, EPSILON);
        double step = 1e-3;
        for (int row = 0; row < input.size(0); row++) {
            for (int col = 0; col < input.size(1); col++) {
                INDArray plus = input.dup();
                INDArray minus = input.dup();
                plus.putScalar(row, col, plus.getDouble(row, col) + step);
                minus.putScalar(row, col, minus.getDouble(row, col) - step);
                double numerical = (weightedOutput(plus, upstream)
                        - weightedOutput(minus, upstream)) / (2 * step);
                assertEquals(numerical, analytical.getDouble(row, col), 2e-3);
            }
        }
    }

    private static double weightedOutput(INDArray input, INDArray upstream) {
        INDArray output = new LayerNormalization(Math.toIntExact(input.size(1)), EPSILON).forward(input);
        double sum = 0;
        for (int row = 0; row < input.size(0); row++) {
            for (int col = 0; col < input.size(1); col++) {
                sum += output.getDouble(row, col) * upstream.getDouble(row, col);
            }
        }
        return sum;
    }

    @Test
    void singleFeatureHasZeroGradient() {
        INDArray input = Nd4j.create(new float[][]{{2}, {7}});
        INDArray upstream = Nd4j.create(new float[][]{{3}, {-4}});
        INDArray gradient = LayerNormalizationGradients.inputGradient(input, upstream, EPSILON);
        assertEquals(0, gradient.norm1Number().doubleValue(), 1e-10);
    }

    @Test
    void constantRowsRemainFinite() {
        INDArray input = Nd4j.create(new float[][]{{2, 2, 2}});
        INDArray upstream = Nd4j.create(new float[][]{{1, 2, 3}});
        INDArray gradient = LayerNormalizationGradients.inputGradient(input, upstream, EPSILON);
        for (int col = 0; col < 3; col++) {
            assertTrue(Double.isFinite(gradient.getDouble(0, col)));
        }
        assertEquals(0, gradient.getDouble(0, 0) + gradient.getDouble(0, 1)
                + gradient.getDouble(0, 2), 1e-6);
    }

    @Test
    void rejectsInvalidInputs() {
        INDArray input = Nd4j.ones(2, 3);
        assertThrows(IllegalArgumentException.class,
                () -> LayerNormalizationGradients.inputGradient(input, Nd4j.ones(3, 2), EPSILON));
        assertThrows(IllegalArgumentException.class,
                () -> LayerNormalizationGradients.inputGradient(input, input, 0));
        assertThrows(NullPointerException.class,
                () -> LayerNormalizationGradients.inputGradient(null, input, EPSILON));
    }
}
