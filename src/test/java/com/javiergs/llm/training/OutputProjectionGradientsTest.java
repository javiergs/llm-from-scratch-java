package com.javiergs.llm.training;

import org.junit.jupiter.api.Test;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import static org.junit.jupiter.api.Assertions.*;

/**
 * Checks analytical gradients against finite-difference approximations.
 *
 * @author Javier Gonzalez-Sanchez
 * @version 0.1.0 (2026-10-31)
 */
class OutputProjectionGradientsTest {
    private static final double STEP = 1e-3;
    private static final double TOLERANCE = 2e-3;

    @Test
    void logitsGradientMatchesFiniteDifferences() {
        INDArray logits = Nd4j.create(new float[][]{{0.2f, -0.4f, 0.6f}, {1.1f, -0.2f, 0.3f}, {0.4f, 0.5f, -0.1f}});
        int[] ids = {0, 2, 1};
        INDArray gradient = OutputProjectionGradients.logitsGradient(logits, ids);
        for (int row = 0; row < 3; row++) {
            for (int column = 0; column < 3; column++) {
                INDArray plus = logits.dup();
                INDArray minus = logits.dup();
                plus.putScalar(row, column, plus.getDouble(row, column) + STEP);
                minus.putScalar(row, column, minus.getDouble(row, column) - STEP);
                double numerical = (NextTokenLoss.mean(plus, ids) - NextTokenLoss.mean(minus, ids)) / (2 * STEP);
                assertEquals(numerical, gradient.getDouble(row, column), TOLERANCE);
            }
        }
        assertEquals(0.0, gradient.getRow(2).norm1Number().doubleValue(), 1e-10);
    }

    @Test
    void weightGradientMatchesFiniteDifferences() {
        INDArray states = Nd4j.create(new float[][]{{0.2f, 0.8f}, {-0.3f, 0.5f}, {0.7f, -0.2f}});
        INDArray weights = Nd4j.create(new float[][]{{0.1f, -0.2f, 0.4f}, {0.3f, 0.5f, -0.1f}});
        int[] ids = {0, 1, 2};
        INDArray gradient = OutputProjectionGradients.weightsGradient(
                states, OutputProjectionGradients.logitsGradient(states.mmul(weights), ids));
        for (int row = 0; row < 2; row++) {
            for (int column = 0; column < 3; column++) {
                INDArray plus = weights.dup();
                INDArray minus = weights.dup();
                plus.putScalar(row, column, plus.getDouble(row, column) + STEP);
                minus.putScalar(row, column, minus.getDouble(row, column) - STEP);
                double numerical = (NextTokenLoss.mean(states.mmul(plus), ids)
                        - NextTokenLoss.mean(states.mmul(minus), ids)) / (2 * STEP);
                assertEquals(numerical, gradient.getDouble(row, column), TOLERANCE);
            }
        }
    }

    @Test
    void hiddenStateGradientMatchesFiniteDifferences() {
        INDArray states = Nd4j.create(new float[][]{{0.2f, 0.8f}, {-0.3f, 0.5f}, {0.7f, -0.2f}});
        INDArray weights = Nd4j.create(new float[][]{{0.1f, -0.2f, 0.4f}, {0.3f, 0.5f, -0.1f}});
        int[] ids = {0, 1, 2};
        INDArray gradient = OutputProjectionGradients.hiddenStatesGradient(
                OutputProjectionGradients.logitsGradient(states.mmul(weights), ids), weights);
        for (int row = 0; row < 3; row++) {
            for (int column = 0; column < 2; column++) {
                INDArray plus = states.dup();
                INDArray minus = states.dup();
                plus.putScalar(row, column, plus.getDouble(row, column) + STEP);
                minus.putScalar(row, column, minus.getDouble(row, column) - STEP);
                double numerical = (NextTokenLoss.mean(plus.mmul(weights), ids)
                        - NextTokenLoss.mean(minus.mmul(weights), ids)) / (2 * STEP);
                assertEquals(numerical, gradient.getDouble(row, column), TOLERANCE);
            }
        }
    }

    @Test
    void rejectsIncompatibleShapes() {
        INDArray a = Nd4j.zeros(2, 3);
        INDArray b = Nd4j.zeros(3, 2);
        assertThrows(IllegalArgumentException.class, () -> OutputProjectionGradients.weightsGradient(a, b));
        assertThrows(IllegalArgumentException.class, () -> OutputProjectionGradients.hiddenStatesGradient(a, b));
    }
}
