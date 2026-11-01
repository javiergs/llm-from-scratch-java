package com.javiergs.llm.training;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

/**
 * Computes the backward pass for a two-layer feed-forward network using
 * the tanh approximation of GELU from FeedForward.
 *
 * @author Javier Gonzalez-Sanchez
 * @version 0.1.0 (2026-10-31)
 */
public final class FeedForwardGradients {
    private FeedForwardGradients() {
    }

    /**
     * Groups derivatives of the feed-forward input and both projection matrices.
     *
     * @param inputGradient derivative with respect to input vectors
     * @param inputWeightsGradient derivative with respect to first projection
     * @param outputWeightsGradient derivative with respect to second projection
     */
    public record Gradients(INDArray inputGradient, INDArray inputWeightsGradient,
                            INDArray outputWeightsGradient) {
    }

    /**
     * Differentiates y = GELU(x * inputWeights) * outputWeights.
     *
     * @param input input token vectors [sequence, dimension]
     * @param inputWeights first projection [dimension, hidden dimension]
     * @param outputWeights second projection [hidden dimension, dimension]
     * @param upstream gradient with respect to y [sequence, dimension]
     * @return gradients for the input and both projection matrices
     */
    public static Gradients backward(INDArray input, INDArray inputWeights,
                                     INDArray outputWeights, INDArray upstream) {
        if (input == null || inputWeights == null || outputWeights == null || upstream == null) {
            throw new NullPointerException("Inputs and gradients must not be null");
        }
        if (input.rank() != 2 || inputWeights.rank() != 2 || outputWeights.rank() != 2
                || upstream.rank() != 2 || input.size(0) == 0 || input.size(1) == 0
                || inputWeights.size(0) != input.size(1)
                || outputWeights.size(0) != inputWeights.size(1)
                || outputWeights.size(1) != input.size(1)
                || upstream.size(0) != input.size(0) || upstream.size(1) != input.size(1)) {
            throw new IllegalArgumentException("Incompatible feed-forward matrix dimensions");
        }
        INDArray preactivation = input.mmul(inputWeights);
        INDArray activated = Nd4j.zeros(preactivation.shape());
        INDArray derivative = Nd4j.zeros(preactivation.shape());
        for (int row = 0; row < preactivation.size(0); row++) {
            for (int col = 0; col < preactivation.size(1); col++) {
                double x = preactivation.getDouble(row, col);
                double k = Math.sqrt(2.0 / Math.PI);
                double t = Math.tanh(k * (x + 0.044715 * x * x * x));
                double y = 0.5 * x * (1.0 + t);
                double dy = 0.5 * (1.0 + t)
                        + 0.5 * x * (1.0 - t * t) * k * (1.0 + 3.0 * 0.044715 * x * x);
                activated.putScalar(row, col, y);
                derivative.putScalar(row, col, dy);
            }
        }
        INDArray outputWeightsGradient = activated.transpose().mmul(upstream);
        INDArray activatedGradient = upstream.mmul(outputWeights.transpose());
        INDArray preactivationGradient = activatedGradient.mul(derivative);
        INDArray inputWeightsGradient = input.transpose().mmul(preactivationGradient);
        INDArray inputGradient = preactivationGradient.mmul(inputWeights.transpose());
        return new Gradients(inputGradient, inputWeightsGradient, outputWeightsGradient);
    }
}
