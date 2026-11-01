package com.javiergs.llm.training;

import com.javiergs.llm.model.GptModel;
import org.nd4j.linalg.api.ndarray.INDArray;

/**
 * Applies a basic stochastic gradient descent (SGD) update to model weights.
 * Gradients must be computed separately by the caller.
 *
 * @author Javier Gonzalez-Sanchez
 * @version 0.1.0 (2026-10-31)
 */
public final class StochasticGradientDescent {
    private final double learningRate;

    /**
     * Creates an SGD optimizer with a constant positive learning rate.
     *
     * @param learningRate finite positive step size
     */
    public StochasticGradientDescent(double learningRate) {
        if (!Double.isFinite(learningRate) || learningRate <= 0) {
            throw new IllegalArgumentException("Learning rate must be finite and positive");
        }
        this.learningRate = learningRate;
    }

    /**
     * Calculates updated copies using weight = weight - learningRate * gradient.
     * All inputs are validated before an update is calculated.
     *
     * @param weights parameter snapshots
     * @param gradients corresponding parameter derivatives
     * @return new weight matrices, leaving inputs unchanged
     */
    public INDArray[] update(INDArray[] weights, INDArray[] gradients) {
        if (weights == null || gradients == null || weights.length == 0
                || weights.length != gradients.length) {
            throw new IllegalArgumentException("Expected matching nonempty parameter and gradient arrays");
        }
        for (int i = 0; i < weights.length; i++) {
            if (weights[i] == null || gradients[i] == null
                    || !java.util.Arrays.equals(weights[i].shape(), gradients[i].shape())) {
                throw new IllegalArgumentException("Gradient shape mismatch at index " + i);
            }
            if (weights[i].rank() != 2) {
                throw new IllegalArgumentException("Expected matrix parameters at index " + i);
            }
            for (long j = 0; j < weights[i].length(); j++) {
                if (!Double.isFinite(weights[i].getDouble(j))
                        || !Double.isFinite(gradients[i].getDouble(j))) {
                    throw new IllegalArgumentException("Non-finite value at index " + i);
                }
            }
        }
        INDArray[] updated = new INDArray[weights.length];
        for (int i = 0; i < weights.length; i++) {
            updated[i] = weights[i].sub(gradients[i].mul(learningRate));
            for (long j = 0; j < updated[i].length(); j++) {
                if (!Double.isFinite(updated[i].getDouble(j))) {
                    throw new IllegalArgumentException("Update produced a non-finite weight at index " + i);
                }
            }
        }
        return updated;
    }

    /**
     * Applies one SGD step to a GPT model using defensive parameter snapshots.
     *
     * @param model model whose weights are updated
     * @param gradients derivatives ordered as in GptModel.parameters()
     */
    public void step(GptModel model, INDArray[] gradients) {
        if (model == null) {
            throw new NullPointerException("model");
        }
        INDArray[] updated = update(model.parameters(), gradients);
        model.setParameters(updated);
    }
}
