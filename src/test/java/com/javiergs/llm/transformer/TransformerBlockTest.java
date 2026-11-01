package com.javiergs.llm.transformer;

import org.junit.jupiter.api.Test;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import static org.junit.jupiter.api.Assertions.*;

/**
 * Tests layer normalization, feed-forward processing, and causal blocks.
 *
 * @author Javier Gonzalez-Sanchez
 * @version 0.1.0 (2026-10-31)
 */
class TransformerBlockTest {

    @Test
    void layerNormalizationCentersEachRow() {
        LayerNormalization norm = new LayerNormalization(3, 1e-5);
        INDArray output = norm.forward(Nd4j.create(new float[][]{{1, 2, 3}, {4, 4, 4}}));
        for (int row = 0; row < 2; row++) {
            double sum = 0;
            for (int col = 0; col < 3; col++) {
                sum += output.getDouble(row, col);
            }
            assertEquals(0.0, sum, 1e-5);
        }
        for (int col = 0; col < 3; col++) {
            assertEquals(0.0, output.getDouble(1, col), 1e-7);
        }
    }

    @Test
    void feedForwardPreservesShape() {
        INDArray output = new FeedForward(4, 16, 2L).forward(Nd4j.ones(3, 4));
        assertArrayEquals(new long[]{3, 4}, output.shape());
    }

    @Test
    void blockPreservesShapeAndInput() {
        TransformerBlock block = new TransformerBlock(4, 2, 16, 42L);
        INDArray input = Nd4j.create(new float[][]{{1, 2, 3, 4}, {2, 3, 4, 5}});
        INDArray before = input.dup();
        assertArrayEquals(new long[]{2, 4}, block.forward(input).shape());
        assertEquals(before, input);
    }

    @Test
    void futureTokenChangesDoNotAffectEarlierOutputs() {
        TransformerBlock block = new TransformerBlock(4, 2, 16, 42L);
        INDArray first = Nd4j.create(new float[][]{{1, 2, 3, 4}, {2, 3, 4, 5}, {3, 4, 5, 6}});
        INDArray second = Nd4j.create(new float[][]{{1, 2, 3, 4}, {2, 3, 4, 5}, {30, 40, 50, 60}});
        INDArray a = block.forward(first);
        INDArray b = block.forward(second);
        for (int row = 0; row < 2; row++) {
            for (int col = 0; col < 4; col++) {
                assertEquals(a.getDouble(row, col), b.getDouble(row, col), 1e-5);
            }
        }
    }

    @Test
    void seedReproducesOutput() {
        INDArray input = Nd4j.ones(3, 4);
        assertEquals(new TransformerBlock(4, 2, 16, 7L).forward(input),
                new TransformerBlock(4, 2, 16, 7L).forward(input));
    }

    @Test
    void rejectsInvalidConfigurationsAndShapes() {
        assertThrows(IllegalArgumentException.class, () -> new TransformerBlock(3, 2, 12, 1L));
        assertThrows(IllegalArgumentException.class, () -> new FeedForward(4, 0, 1L));
        assertThrows(IllegalArgumentException.class, () -> new LayerNormalization(0, 1e-5));
        assertThrows(IllegalArgumentException.class, () -> new LayerNormalization(4, 0));
        assertThrows(IllegalArgumentException.class,
                () -> new TransformerBlock(4, 2, 16, 1L).forward(Nd4j.ones(2, 3)));
    }
}
