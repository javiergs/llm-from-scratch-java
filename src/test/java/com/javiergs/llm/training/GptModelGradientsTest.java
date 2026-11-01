package com.javiergs.llm.training;

import com.javiergs.llm.model.GptModel;
import org.junit.jupiter.api.Test;
import org.nd4j.linalg.api.ndarray.INDArray;
import static org.junit.jupiter.api.Assertions.*;

/**
 * Checks complete GPT gradients and one genuine next-token training step.
 *
 * @author Javier Gonzalez-Sanchez
 * @version 0.1.0 (2026-10-31)
 */
class GptModelGradientsTest {
    @Test
    void producesMatchingFiniteGradientsForAllModelParameters() {
        GptModel model = new GptModel(4, 4, 4, 2, 1, 6, 42);
        int[] sequence = {0, 1, 0, 2};
        INDArray[] weights = model.parameters();
        INDArray[] gradients = GptModelGradients.backward(model, sequence);
        assertEquals(weights.length, gradients.length);
        for (int i = 0; i < weights.length; i++) {
            assertArrayEquals(weights[i].shape(), gradients[i].shape());
            for (long j = 0; j < gradients[i].length(); j++) {
                assertTrue(Double.isFinite(gradients[i].getDouble(j)));
            }
        }
        // Check representative entries from embeddings, attention, FFN, and output.
        int[] indices = {0, 1, 2, 3, 4, 5, 6, 7, 8};
        for (int index : indices) {
            INDArray[] plus = model.parameters();
            INDArray[] minus = model.parameters();
            double step = 1e-3;
            plus[index].putScalar(0, 0, plus[index].getDouble(0, 0) + step);
            minus[index].putScalar(0, 0, minus[index].getDouble(0, 0) - step);
            model.setParameters(plus);
            double high = GptModelGradients.loss(model, sequence);
            model.setParameters(minus);
            double low = GptModelGradients.loss(model, sequence);
            model.setParameters(weights);
            assertEquals((high - low) / (2 * step),
                    gradients[index].getDouble(0, 0), 1e-2, "Parameter " + index);
        }
    }

    @Test
    void oneSgdStepCanReduceActualNextTokenLoss() {
        GptModel model = new GptModel(4, 4, 4, 2, 1, 6, 42);
        int[] sequence = {0, 1, 0, 1};
        double before = GptModelGradients.loss(model, sequence);
        INDArray[] gradients = GptModelGradients.backward(model, sequence);
        new StochasticGradientDescent(0.01).step(model, gradients);
        double after = GptModelGradients.loss(model, sequence);
        assertTrue(after < before, "A small SGD step should decrease next-token loss");
    }

    @Test
    void requiresAtLeastTwoTokens() {
        GptModel model = new GptModel(4, 4, 4, 2, 1, 6, 42);
        assertThrows(IllegalArgumentException.class,
                () -> GptModelGradients.backward(model, new int[]{0}));
    }
}
