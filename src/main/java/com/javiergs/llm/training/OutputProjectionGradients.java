package com.javiergs.llm.training;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

/**
 * Computes gradients for next-token loss and a linear vocabulary projection.
 * This lesson does not yet update model parameters or backpropagate through
 * the transformer blocks.
 *
 * @author Javier Gonzalez-Sanchez
 * @version 0.1.0 (2026-10-31)
 */
public final class OutputProjectionGradients {
    private OutputProjectionGradients() {
    }

    /**
     * Differentiates mean next-token cross-entropy with respect to logits.
     * The final logit row has zero gradient because it has no next-token target.
     *
     * @param logits unnormalized scores [sequence length, vocabulary size]
     * @param tokenIds token IDs whose successors are prediction targets
     * @return gradient with the same shape as logits
     */
    public static INDArray logitsGradient(INDArray logits, int[] tokenIds) {
        // Reuse the loss function's input and target validation.
        NextTokenLoss.mean(logits, tokenIds);
        int positions = tokenIds.length - 1;
        int vocabularySize = Math.toIntExact(logits.size(1));
        INDArray gradient = Nd4j.zeros(logits.shape());
        for (int row = 0; row < positions; row++) {
            double maximum = Double.NEGATIVE_INFINITY;
            for (int column = 0; column < vocabularySize; column++) {
                maximum = Math.max(maximum, logits.getDouble(row, column));
            }
            double denominator = 0.0;
            for (int column = 0; column < vocabularySize; column++) {
                denominator += Math.exp(logits.getDouble(row, column) - maximum);
            }
            for (int column = 0; column < vocabularySize; column++) {
                double probability = Math.exp(logits.getDouble(row, column) - maximum) / denominator;
                double targetIndicator = column == tokenIds[row + 1] ? 1.0 : 0.0;
                gradient.putScalar(row, column, (probability - targetIndicator) / positions);
            }
        }
        return gradient;
    }

    /**
     * Differentiates logits = hiddenStates * weights with respect to weights.
     *
     * @param hiddenStates transformer outputs [sequence length, model dimension]
     * @param logitsGradient gradient [sequence length, vocabulary size]
     * @return weight gradient [model dimension, vocabulary size]
     */
    public static INDArray weightsGradient(INDArray hiddenStates, INDArray logitsGradient) {
        if (hiddenStates == null || logitsGradient == null) {
            throw new NullPointerException("hiddenStates and logitsGradient must not be null");
        }
        if (hiddenStates.rank() != 2 || logitsGradient.rank() != 2
                || hiddenStates.size(0) != logitsGradient.size(0)) {
            throw new IllegalArgumentException("Both inputs must be matrices with matching row counts");
        }
        return hiddenStates.transpose().mmul(logitsGradient);
    }

    /**
     * Differentiates logits = hiddenStates * weights with respect to hidden states.
     *
     * @param logitsGradient gradient [sequence length, vocabulary size]
     * @param weights projection weights [model dimension, vocabulary size]
     * @return hidden-state gradient [sequence length, model dimension]
     */
    public static INDArray hiddenStatesGradient(INDArray logitsGradient, INDArray weights) {
        if (logitsGradient == null || weights == null) {
            throw new NullPointerException("logitsGradient and weights must not be null");
        }
        if (logitsGradient.rank() != 2 || weights.rank() != 2
                || logitsGradient.size(1) != weights.size(1)) {
            throw new IllegalArgumentException("Both inputs must be matrices with matching vocabulary dimensions");
        }
        return logitsGradient.mmul(weights.transpose());
    }
}
