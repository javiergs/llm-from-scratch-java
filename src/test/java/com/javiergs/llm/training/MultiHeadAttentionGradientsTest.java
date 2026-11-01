package com.javiergs.llm.training;

import org.junit.jupiter.api.Test;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.NDArrayIndex;
import static org.junit.jupiter.api.Assertions.*;

/**
 * Checks multi-head attention backward gradients against finite differences.
 *
 * @author Javier Gonzalez-Sanchez
 * @version 0.1.0 (2026-10-31)
 */
class MultiHeadAttentionGradientsTest {
    private final INDArray input = Nd4j.create(new float[][]{{0.2f, -0.3f, 0.1f, 0.5f},
            {0.7f, 0.2f, -0.4f, 0.1f}, {-0.1f, 0.6f, 0.3f, -0.2f}});
    private final INDArray q = matrix(0.10f);
    private final INDArray k = matrix(-0.07f);
    private final INDArray v = matrix(0.04f);
    private final INDArray o = matrix(0.13f);
    private final INDArray upstream = Nd4j.create(new float[][]{{0.2f, 0.1f, -0.3f, 0.5f},
            {-0.4f, 0.2f, 0.6f, 0.1f}, {0.3f, -0.1f, 0.2f, 0.4f}});

    private static INDArray matrix(float scale) {
        float[][] data = new float[4][4];
        for (int row = 0; row < 4; row++) {
            for (int col = 0; col < 4; col++) {
                data[row][col] = scale * (1 + row * 2 - col);
            }
        }
        return Nd4j.create(data);
    }

    @Test
    void allFiveGradientsMatchNumericalDerivatives() {
        MultiHeadAttentionGradients.Gradients g =
                MultiHeadAttentionGradients.backward(input, q, k, v, o, 2, upstream);
        check(input, g.inputGradient(), 0);
        check(q, g.queryWeightsGradient(), 1);
        check(k, g.keyWeightsGradient(), 2);
        check(v, g.valueWeightsGradient(), 3);
        check(o, g.outputWeightsGradient(), 4);
    }

    private void check(INDArray parameter, INDArray analytical, int selected) {
        double step = 1e-3;
        for (int row = 0; row < parameter.size(0); row++) {
            for (int col = 0; col < parameter.size(1); col++) {
                INDArray plus = parameter.dup();
                INDArray minus = parameter.dup();
                plus.putScalar(row, col, plus.getDouble(row, col) + step);
                minus.putScalar(row, col, minus.getDouble(row, col) - step);
                double high = objective(selected == 0 ? plus : input, selected == 1 ? plus : q,
                        selected == 2 ? plus : k, selected == 3 ? plus : v, selected == 4 ? plus : o);
                double low = objective(selected == 0 ? minus : input, selected == 1 ? minus : q,
                        selected == 2 ? minus : k, selected == 3 ? minus : v, selected == 4 ? minus : o);
                assertEquals((high - low) / (2 * step), analytical.getDouble(row, col), 3e-3);
            }
        }
    }

    private double objective(INDArray x, INDArray wq, INDArray wk, INDArray wv, INDArray wo) {
        INDArray queries = x.mmul(wq);
        INDArray keys = x.mmul(wk);
        INDArray values = x.mmul(wv);
        INDArray concatenated = Nd4j.zeros(x.size(0), x.size(1));
        for (int head = 0; head < 2; head++) {
            int start = head * 2;
            INDArray query = queries.get(NDArrayIndex.all(), NDArrayIndex.interval(start, start + 2));
            INDArray key = keys.get(NDArrayIndex.all(), NDArrayIndex.interval(start, start + 2));
            INDArray value = values.get(NDArrayIndex.all(), NDArrayIndex.interval(start, start + 2));
            INDArray scores = query.mmul(key.transpose()).div(Math.sqrt(2));
            INDArray probabilities = Nd4j.zeros(x.size(0), x.size(0));
            for (int row = 0; row < x.size(0); row++) {
                double max = Double.NEGATIVE_INFINITY;
                for (int col = 0; col <= row; col++) {
                    max = Math.max(max, scores.getDouble(row, col));
                }
                double sum = 0;
                for (int col = 0; col <= row; col++) {
                    sum += Math.exp(scores.getDouble(row, col) - max);
                }
                for (int col = 0; col <= row; col++) {
                    probabilities.putScalar(row, col, Math.exp(scores.getDouble(row, col) - max) / sum);
                }
            }
            concatenated.get(NDArrayIndex.all(), NDArrayIndex.interval(start, start + 2))
                    .assign(probabilities.mmul(value));
        }
        INDArray output = concatenated.mmul(wo);
        double result = 0;
        for (int row = 0; row < output.size(0); row++) {
            for (int col = 0; col < output.size(1); col++) {
                result += output.getDouble(row, col) * upstream.getDouble(row, col);
            }
        }
        return result;
    }

    @Test
    void rejectsInvalidHeadCounts() {
        assertThrows(IllegalArgumentException.class,
                () -> MultiHeadAttentionGradients.backward(input, q, k, v, o, 3, upstream));
    }
}
