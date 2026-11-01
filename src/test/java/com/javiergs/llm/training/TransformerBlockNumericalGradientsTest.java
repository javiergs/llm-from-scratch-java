package com.javiergs.llm.training;

import com.javiergs.llm.transformer.LayerNormalization;
import org.junit.jupiter.api.Test;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.NDArrayIndex;

import static org.junit.jupiter.api.Assertions.assertEquals;

/**
 * Checks the complete pre-norm block chain rule using numerical derivatives.
 *
 * @author Javier Gonzalez-Sanchez
 * @version 0.1.0 (2026-10-31)
 */
class TransformerBlockNumericalGradientsTest {
    private static final double EPSILON = 1e-5;
    private static final double STEP = 1e-3;

    @Test
    void inputAndAllSixProjectionGradientsMatchFiniteDifferences() {
        INDArray input = Nd4j.create(new float[][]{
                {0.2f, -0.5f, 0.7f, 0.1f},
                {-0.3f, 0.8f, -0.2f, 0.4f}
        });
        INDArray upstream = Nd4j.create(new float[][]{
                {0.3f, -0.2f, 0.1f, 0.5f},
                {-0.1f, 0.4f, 0.2f, -0.3f}
        });
        INDArray[] weights = {
                matrix(4, 4, 0.09), matrix(4, 4, -0.07),
                matrix(4, 4, 0.05), matrix(4, 4, 0.11),
                matrix(4, 5, 0.08), matrix(5, 4, -0.06)
        };
        TransformerBlockGradients.Gradients gradients = TransformerBlockGradients.backward(
                input, parameters(weights), 2, EPSILON, upstream);
        INDArray[] expected = {
                gradients.inputGradient(),
                gradients.attention().queryWeightsGradient(),
                gradients.attention().keyWeightsGradient(),
                gradients.attention().valueWeightsGradient(),
                gradients.attention().outputWeightsGradient(),
                gradients.feedForward().inputWeightsGradient(),
                gradients.feedForward().outputWeightsGradient()
        };
        for (int selected = 0; selected < expected.length; selected++) {
            INDArray parameter = selected == 0 ? input : weights[selected - 1];
            for (int row = 0; row < parameter.size(0); row++) {
                for (int col = 0; col < parameter.size(1); col++) {
                    double original = parameter.getDouble(row, col);
                    parameter.putScalar(row, col, original + STEP);
                    double high = objective(input, weights, upstream);
                    parameter.putScalar(row, col, original - STEP);
                    double low = objective(input, weights, upstream);
                    parameter.putScalar(row, col, original);
                    assertEquals((high - low) / (2 * STEP),
                            expected[selected].getDouble(row, col), 5e-3,
                            "Gradient mismatch for parameter " + selected + " [" + row + "," + col + "]");
                }
            }
        }
    }

    private static INDArray matrix(int rows, int columns, double scale) {
        INDArray result = Nd4j.zeros(rows, columns);
        for (int row = 0; row < rows; row++) {
            for (int col = 0; col < columns; col++) {
                result.putScalar(row, col, scale * (1 + row * 0.7 - col * 0.4));
            }
        }
        return result;
    }

    private static TransformerBlockGradients.Weights parameters(INDArray[] w) {
        return new TransformerBlockGradients.Weights(w[0], w[1], w[2], w[3], w[4], w[5]);
    }

    private static double objective(INDArray input, INDArray[] w, INDArray upstream) {
        INDArray normalized = normalize(input);
        INDArray q = normalized.mmul(w[0]);
        INDArray k = normalized.mmul(w[1]);
        INDArray v = normalized.mmul(w[2]);
        INDArray heads = Nd4j.zeros(input.size(0), input.size(1));
        for (int head = 0; head < 2; head++) {
            int start = head * 2;
            INDArray query = slice(q, start);
            INDArray key = slice(k, start);
            INDArray value = slice(v, start);
            INDArray scores = query.mmul(key.transpose()).div(Math.sqrt(2));
            INDArray probabilities = Nd4j.zeros(input.size(0), input.size(0));
            for (int row = 0; row < input.size(0); row++) {
                double maximum = Double.NEGATIVE_INFINITY;
                for (int col = 0; col <= row; col++) {
                    maximum = Math.max(maximum, scores.getDouble(row, col));
                }
                double sum = 0;
                for (int col = 0; col <= row; col++) {
                    sum += Math.exp(scores.getDouble(row, col) - maximum);
                }
                for (int col = 0; col <= row; col++) {
                    probabilities.putScalar(row, col,
                            Math.exp(scores.getDouble(row, col) - maximum) / sum);
                }
            }
            heads.get(NDArrayIndex.all(), NDArrayIndex.interval(start, start + 2))
                    .assign(probabilities.mmul(value));
        }
        INDArray residual = input.add(heads.mmul(w[3]));
        INDArray hidden = normalize(residual).mmul(w[4]);
        INDArray activated = Nd4j.zeros(hidden.shape());
        for (int row = 0; row < hidden.size(0); row++) {
            for (int col = 0; col < hidden.size(1); col++) {
                double x = hidden.getDouble(row, col);
                activated.putScalar(row, col, 0.5 * x * (1 + Math.tanh(
                        Math.sqrt(2.0 / Math.PI) * (x + 0.044715 * x * x * x))));
            }
        }
        INDArray output = residual.add(activated.mmul(w[5]));
        double loss = 0;
        for (int row = 0; row < output.size(0); row++) {
            for (int col = 0; col < output.size(1); col++) {
                loss += output.getDouble(row, col) * upstream.getDouble(row, col);
            }
        }
        return loss;
    }

    private static INDArray normalize(INDArray input) {
        return new LayerNormalization((int) input.size(1), EPSILON).forward(input);
    }

    private static INDArray slice(INDArray matrix, int start) {
        return matrix.get(NDArrayIndex.all(), NDArrayIndex.interval(start, start + 2));
    }
}
