package com.javiergs.llm.training;

import com.javiergs.llm.model.GptModel;
import com.javiergs.llm.transformer.LayerNormalization;
import org.nd4j.linalg.api.ndarray.INDArray;

/**
 * Connects next-token loss to all trainable GPT parameters using the
 * analytical backward passes developed in earlier lessons.
 *
 * @author Javier Gonzalez-Sanchez
 * @version 0.1.0 (2026-10-31)
 */
public final class GptModelGradients {
    private static final double EPSILON = 1e-5;

    private GptModelGradients() {
    }

    /**
     * Computes gradients in exactly the order returned by model.parameters().
     * The final logit row is excluded from next-token cross-entropy.
     *
     * @param model decoder-only model
     * @param tokenIds sequence containing at least two tokens
     * @return one gradient matrix per trainable parameter
     */
    public static INDArray[] backward(GptModel model, int[] tokenIds) {
        if (model == null || tokenIds == null) {
            throw new NullPointerException("model and tokenIds must not be null");
        }
        if (tokenIds.length < 2) {
            throw new IllegalArgumentException("Next-token training requires at least two tokens");
        }
        INDArray[] weights = model.parameters();
        int blocks = (weights.length - 3) / 6;
        INDArray[] states = model.hiddenStates(tokenIds);
        INDArray last = states[blocks];
        INDArray normalized = new LayerNormalization(model.dimension(), EPSILON).forward(last);
        INDArray logits = normalized.mmul(weights[weights.length - 1]);
        INDArray dLogits = OutputProjectionGradients.logitsGradient(logits, tokenIds);
        INDArray[] gradients = new INDArray[weights.length];
        gradients[gradients.length - 1] =
                OutputProjectionGradients.weightsGradient(normalized, dLogits);
        INDArray dNormalized = OutputProjectionGradients.hiddenStatesGradient(
                dLogits, weights[weights.length - 1]);
        INDArray upstream = LayerNormalizationGradients.inputGradient(
                last, dNormalized, EPSILON);
        for (int block = blocks - 1; block >= 0; block--) {
            int offset = 2 + 6 * block;
            TransformerBlockGradients.Weights blockWeights = new TransformerBlockGradients.Weights(
                    weights[offset], weights[offset + 1], weights[offset + 2],
                    weights[offset + 3], weights[offset + 4], weights[offset + 5]);
            TransformerBlockGradients.Gradients result = TransformerBlockGradients.backward(
                    states[block], blockWeights, model.heads(), EPSILON, upstream);
            gradients[offset] = result.attention().queryWeightsGradient();
            gradients[offset + 1] = result.attention().keyWeightsGradient();
            gradients[offset + 2] = result.attention().valueWeightsGradient();
            gradients[offset + 3] = result.attention().outputWeightsGradient();
            gradients[offset + 4] = result.feedForward().inputWeightsGradient();
            gradients[offset + 5] = result.feedForward().outputWeightsGradient();
            upstream = result.inputGradient();
        }
        gradients[0] = EmbeddingGradients.tokenWeights(tokenIds, upstream, (int) weights[0].size(0));
        gradients[1] = EmbeddingGradients.positionWeights(tokenIds, upstream, (int) weights[1].size(0));
        return gradients;
    }

    /**
     * Computes next-token loss using the model's current parameters.
     *
     * @param model decoder-only model
     * @param tokenIds sequence containing next-token targets
     * @return mean cross-entropy
     */
    public static double loss(GptModel model, int[] tokenIds) {
        return NextTokenLoss.mean(model.forward(tokenIds), tokenIds);
    }
}
