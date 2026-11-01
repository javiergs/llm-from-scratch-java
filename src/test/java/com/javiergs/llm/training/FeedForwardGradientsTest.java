package com.javiergs.llm.training;

import org.junit.jupiter.api.Test;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import static org.junit.jupiter.api.Assertions.*;

/**
 * Validates feed-forward projection and GELU gradients numerically.
 *
 * @author Javier Gonzalez-Sanchez
 * @version 0.1.0 (2026-10-31)
 */
class FeedForwardGradientsTest {
    private final INDArray input = Nd4j.create(new float[][]{{0.2f, -0.4f}, {0.7f, 0.1f}});
    private final INDArray first = Nd4j.create(new float[][]{{0.3f, -0.2f, 0.1f}, {-0.4f, 0.6f, 0.2f}});
    private final INDArray second = Nd4j.create(new float[][]{{0.5f, -0.3f}, {0.2f, 0.1f}, {-0.6f, 0.4f}});
    private final INDArray upstream = Nd4j.create(new float[][]{{0.2f, -0.3f}, {0.5f, 0.7f}});

    @Test
    void matchesFiniteDifferencesForAllThreeGradients() {
        FeedForwardGradients.Gradients gradients = FeedForwardGradients.backward(input, first, second, upstream);
        checkGradient(input, gradients.inputGradient(), 0);
        checkGradient(first, gradients.inputWeightsGradient(), 1);
        checkGradient(second, gradients.outputWeightsGradient(), 2);
    }

    private void checkGradient(INDArray parameter, INDArray analytical, int selected) {
        double step = 1e-3;
        for (int row = 0; row < parameter.size(0); row++) {
            for (int col = 0; col < parameter.size(1); col++) {
                INDArray plus = parameter.dup();
                INDArray minus = parameter.dup();
                plus.putScalar(row, col, plus.getDouble(row, col) + step);
                minus.putScalar(row, col, minus.getDouble(row, col) - step);
                double high = objective(selected == 0 ? plus : input,
                        selected == 1 ? plus : first, selected == 2 ? plus : second);
                double low = objective(selected == 0 ? minus : input,
                        selected == 1 ? minus : first, selected == 2 ? minus : second);
                assertEquals((high - low) / (2 * step), analytical.getDouble(row, col), 2e-3);
            }
        }
    }

    private double objective(INDArray x, INDArray w1, INDArray w2) {
        INDArray z = x.mmul(w1);
        INDArray a = Nd4j.zeros(z.shape());
        for (int row = 0; row < z.size(0); row++) {
            for (int col = 0; col < z.size(1); col++) {
                double v = z.getDouble(row, col);
                a.putScalar(row, col, 0.5 * v * (1.0
                        + Math.tanh(Math.sqrt(2.0 / Math.PI) * (v + 0.044715 * v * v * v))));
            }
        }
        INDArray output = a.mmul(w2);
        double total = 0;
        for (int row = 0; row < output.size(0); row++) {
            for (int col = 0; col < output.size(1); col++) {
                total += output.getDouble(row, col) * upstream.getDouble(row, col);
            }
        }
        return total;
    }

    @Test
    void rejectsIncompatibleShapes() {
        assertThrows(IllegalArgumentException.class,
                () -> FeedForwardGradients.backward(input, second, first, upstream));
        assertThrows(NullPointerException.class,
                () -> FeedForwardGradients.backward(null, first, second, upstream));
    }
}
