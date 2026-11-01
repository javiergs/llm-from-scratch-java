package com.javiergs.llm.training;

import org.junit.jupiter.api.Test;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import static org.junit.jupiter.api.Assertions.*;

/**
 * Verifies single-head causal attention gradients using finite differences.
 *
 * @author Javier Gonzalez-Sanchez
 * @version 0.1.0 (2026-10-31)
 */
class CausalAttentionGradientsTest {
    private final INDArray input = Nd4j.create(new float[][]{{0.2f, -0.4f}, {0.7f, 0.1f}, {-0.3f, 0.6f}});
    private final INDArray wq = Nd4j.create(new float[][]{{0.3f, -0.2f}, {0.1f, 0.4f}});
    private final INDArray wk = Nd4j.create(new float[][]{{-0.4f, 0.2f}, {0.3f, 0.5f}});
    private final INDArray wv = Nd4j.create(new float[][]{{0.2f, 0.6f}, {-0.1f, 0.3f}});
    private final INDArray upstream = Nd4j.create(new float[][]{{0.2f, -0.3f}, {0.5f, 0.7f}, {-0.1f, 0.4f}});

    @Test
    void matchesFiniteDifferencesForInputAndAllProjectionWeights() {
        CausalAttentionGradients.Gradients gradients =
                CausalAttentionGradients.backward(input, wq, wk, wv, upstream);
        check(input, gradients.inputGradient(), 0);
        check(wq, gradients.queryWeightsGradient(), 1);
        check(wk, gradients.keyWeightsGradient(), 2);
        check(wv, gradients.valueWeightsGradient(), 3);
    }

    private void check(INDArray parameter, INDArray analytical, int selected) {
        double step = 1e-3;
        for (int row = 0; row < parameter.size(0); row++) {
            for (int col = 0; col < parameter.size(1); col++) {
                INDArray plus = parameter.dup();
                INDArray minus = parameter.dup();
                plus.putScalar(row, col, plus.getDouble(row, col) + step);
                minus.putScalar(row, col, minus.getDouble(row, col) - step);
                double high = objective(selected == 0 ? plus : input, selected == 1 ? plus : wq,
                        selected == 2 ? plus : wk, selected == 3 ? plus : wv);
                double low = objective(selected == 0 ? minus : input, selected == 1 ? minus : wq,
                        selected == 2 ? minus : wk, selected == 3 ? minus : wv);
                assertEquals((high - low) / (2 * step), analytical.getDouble(row, col), 2e-3);
            }
        }
    }

    private double objective(INDArray x, INDArray query, INDArray key, INDArray value) {
        INDArray q = x.mmul(query);
        INDArray k = x.mmul(key);
        INDArray v = x.mmul(value);
        INDArray scores = q.mmul(k.transpose()).div(Math.sqrt(x.size(1)));
        INDArray p = Nd4j.zeros(x.size(0), x.size(0));
        for (int row = 0; row < x.size(0); row++) {
            double maximum = Double.NEGATIVE_INFINITY;
            for (int col = 0; col <= row; col++) {
                maximum = Math.max(maximum, scores.getDouble(row, col));
            }
            double sum = 0;
            for (int col = 0; col <= row; col++) {
                sum += Math.exp(scores.getDouble(row, col) - maximum);
            }
            for (int col = 0; col <= row; col++) {
                p.putScalar(row, col, Math.exp(scores.getDouble(row, col) - maximum) / sum);
            }
        }
        INDArray output = p.mmul(v);
        double result = 0;
        for (int row = 0; row < output.size(0); row++) {
            for (int col = 0; col < output.size(1); col++) {
                result += output.getDouble(row, col) * upstream.getDouble(row, col);
            }
        }
        return result;
    }

    @Test
    void firstPositionCannotDependOnFutureTokens() {
        INDArray firstOnly = Nd4j.create(new float[][]{{1, 0}, {0, 0}, {0, 0}});
        CausalAttentionGradients.Gradients gradients =
                CausalAttentionGradients.backward(input, wq, wk, wv, firstOnly);
        for (int row = 1; row < input.size(0); row++) {
            for (int col = 0; col < input.size(1); col++) {
                assertEquals(0, gradients.inputGradient().getDouble(row, col), 1e-7);
            }
        }
    }

    @Test
    void rejectsInvalidShapes() {
        assertThrows(IllegalArgumentException.class,
                () -> CausalAttentionGradients.backward(input, wq, wk, Nd4j.ones(3, 2), upstream));
        assertThrows(NullPointerException.class,
                () -> CausalAttentionGradients.backward(null, wq, wk, wv, upstream));
    }
}
