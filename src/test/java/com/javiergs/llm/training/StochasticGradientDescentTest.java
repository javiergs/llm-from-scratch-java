package com.javiergs.llm.training;

import com.javiergs.llm.model.GptModel;
import org.junit.jupiter.api.Test;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import static org.junit.jupiter.api.Assertions.*;

/**
 * Tests SGD arithmetic, validation, and explicit model weight updates.
 *
 * @author Javier Gonzalez-Sanchez
 * @version 0.1.0 (2026-10-31)
 */
class StochasticGradientDescentTest {
    @Test
    void movesWeightsTowardMinimumOfQuadraticObjective() {
        INDArray weights = Nd4j.create(new float[][]{{2f, -4f}});
        INDArray gradients = weights.dup(); // d(0.5 * sum(weights^2))/dweights = weights
        StochasticGradientDescent optimizer = new StochasticGradientDescent(0.25);
        INDArray updated = optimizer.update(new INDArray[]{weights},
                new INDArray[]{gradients})[0];
        assertEquals(1.5, updated.getDouble(0, 0), 1e-6);
        assertEquals(-3, updated.getDouble(0, 1), 1e-6);
        assertEquals(2, weights.getDouble(0, 0), 1e-6);
        assertTrue(updated.mul(updated).sumNumber().doubleValue()
                < weights.mul(weights).sumNumber().doubleValue());
    }

    @Test
    void stepUpdatesModelParameters() {
        GptModel model = new GptModel(5, 4, 4, 2, 1, 6, 42);
        INDArray[] before = model.parameters();
        INDArray[] gradients = new INDArray[before.length];
        for (int i = 0; i < before.length; i++) {
            gradients[i] = Nd4j.ones(before[i].shape());
        }
        new StochasticGradientDescent(0.1).step(model, gradients);
        INDArray[] after = model.parameters();
        for (int i = 0; i < before.length; i++) {
            for (long j = 0; j < before[i].length(); j++) {
                assertEquals(before[i].getDouble(j) - 0.1, after[i].getDouble(j), 1e-5);
            }
        }
    }

    @Test
    void invalidGradientsDoNotPartiallyUpdateModel() {
        GptModel model = new GptModel(5, 4, 4, 2, 1, 6, 42);
        INDArray[] before = model.parameters();
        INDArray[] gradients = new INDArray[before.length];
        for (int i = 0; i < before.length; i++) {
            gradients[i] = Nd4j.zeros(before[i].shape());
        }
        gradients[gradients.length - 1] = Nd4j.zeros(1, 1);
        assertThrows(IllegalArgumentException.class,
                () -> new StochasticGradientDescent(0.1).step(model, gradients));
        INDArray[] after = model.parameters();
        for (int i = 0; i < before.length; i++) {
            assertEquals(0, before[i].sub(after[i]).norm2Number().doubleValue(), 1e-6);
        }
    }

    @Test
    void rejectsInvalidLearningRatesAndNonFiniteGradients() {
        assertThrows(IllegalArgumentException.class, () -> new StochasticGradientDescent(0));
        assertThrows(IllegalArgumentException.class,
                () -> new StochasticGradientDescent(Double.NaN));
        assertThrows(IllegalArgumentException.class,
                () -> new StochasticGradientDescent(Double.POSITIVE_INFINITY));
        INDArray gradient = Nd4j.create(new float[][]{{Float.NaN}});
        assertThrows(IllegalArgumentException.class,
                () -> new StochasticGradientDescent(0.1).update(
                        new INDArray[]{Nd4j.ones(1, 1)}, new INDArray[]{gradient}));
    }
}
