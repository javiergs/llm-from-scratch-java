package com.javiergs.llm.model;

import org.junit.jupiter.api.Test;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Verifies defensive parameter snapshots and controlled model updates.
 *
 * @author Javier Gonzalez-Sanchez
 * @version 0.1.0 (2026-10-31)
 */
class GptModelParametersTest {
    @Test
    void parameterCopiesDoNotMutateTheModelAndExplicitUpdatesDo() {
        GptModel model = new GptModel(5, 4, 4, 2, 2, 6, 42);
        int[] tokens = {0, 1, 2};
        INDArray original = model.forward(tokens).dup();
        INDArray[] weights = model.parameters();
        assertEquals(15, weights.length);
        weights[weights.length - 1].assign(0);
        // Snapshot modifications are isolated until explicitly applied.
        assertEquals(0, original.sub(model.forward(tokens)).norm2Number().doubleValue(), 1e-6);
        model.setParameters(weights);
        assertEquals(0, model.forward(tokens).norm2Number().doubleValue(), 1e-6);
        assertTrue(original.norm2Number().doubleValue() > 0);
    }

    @Test
    void rejectsInvalidShapesWithoutChangingAnyWeights() {
        GptModel model = new GptModel(5, 4, 4, 2, 1, 6, 42);
        INDArray[] weights = model.parameters();
        INDArray[] before = model.parameters();
        weights[weights.length - 1] = Nd4j.zeros(1, 1);
        assertThrows(IllegalArgumentException.class, () -> model.setParameters(weights));
        INDArray[] after = model.parameters();
        for (int i = 0; i < before.length; i++) {
            assertEquals(0, before[i].sub(after[i]).norm2Number().doubleValue(), 1e-6);
        }
    }
}
