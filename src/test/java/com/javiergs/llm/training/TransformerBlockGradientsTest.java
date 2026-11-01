package com.javiergs.llm.training;

import org.junit.jupiter.api.Test;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import static org.junit.jupiter.api.Assertions.*;

/**
 * Checks gradient propagation through the transformer's residual paths.
 *
 * @author Javier Gonzalez-Sanchez
 * @version 0.1.0 (2026-10-31)
 */
class TransformerBlockGradientsTest {
    @Test
    void zeroProjectionsPreserveUpstreamGradientThroughResiduals() {
        INDArray input = Nd4j.create(new float[][]{{0.2f, 0.4f}, {-0.3f, 0.7f}});
        INDArray upstream = Nd4j.create(new float[][]{{0.1f, -0.2f}, {0.5f, 0.3f}});
        INDArray zero = Nd4j.zeros(2, 2);
        INDArray hiddenZero = Nd4j.zeros(2, 3);
        INDArray outputZero = Nd4j.zeros(3, 2);
        TransformerBlockGradients.Weights weights = new TransformerBlockGradients.Weights(
                zero, zero, zero, zero, hiddenZero, outputZero);
        TransformerBlockGradients.Gradients gradients =
                TransformerBlockGradients.backward(input, weights, 1, 1e-5, upstream);
        for (int row = 0; row < 2; row++) {
            for (int col = 0; col < 2; col++) {
                assertEquals(upstream.getDouble(row, col),
                        gradients.inputGradient().getDouble(row, col), 1e-6);
            }
        }
    }

    @Test
    void rejectsInvalidHeadCount() {
        INDArray input = Nd4j.ones(2, 2);
        INDArray zero = Nd4j.zeros(2, 2);
        TransformerBlockGradients.Weights weights =
                new TransformerBlockGradients.Weights(zero, zero, zero, zero, zero, zero);
        assertThrows(IllegalArgumentException.class,
                () -> TransformerBlockGradients.backward(input, weights, 3, 1e-5, input));
    }
}
