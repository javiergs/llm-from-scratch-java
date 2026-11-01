package com.javiergs.llm.model;

import org.junit.jupiter.api.Test;
import org.nd4j.linalg.api.ndarray.INDArray;
import static org.junit.jupiter.api.Assertions.*;

/**
 * Tests model output shapes, causal independence, and validation.
 *
 * @author Javier Gonzalez-Sanchez
 * @version 0.1.0 (2026-10-31)
 */
class GptModelTest {

    @Test
    void producesVocabularyLogitsAtEveryPosition() {
        GptModel model = new GptModel(7, 8, 4, 2, 2, 16, 42L);
        assertArrayEquals(new long[]{3, 7}, model.forward(new int[]{1, 2, 3}).shape());
        assertEquals(7, model.vocabularySize());
        assertEquals(4, model.dimension());
    }

    @Test
    void preservesCausalityThroughStackedBlocks() {
        GptModel model = new GptModel(6, 8, 4, 2, 2, 16, 42L);
        INDArray a = model.forward(new int[]{1, 2, 3});
        INDArray b = model.forward(new int[]{1, 2, 4});
        for (int row = 0; row < 2; row++) {
            for (int col = 0; col < 6; col++) {
                assertEquals(a.getDouble(row, col), b.getDouble(row, col), 1e-5);
            }
        }
    }

    @Test
    void seededInitializationIsReproducible() {
        INDArray a = new GptModel(5, 8, 4, 2, 2, 16, 9L).forward(new int[]{1, 2});
        INDArray b = new GptModel(5, 8, 4, 2, 2, 16, 9L).forward(new int[]{1, 2});
        assertEquals(a, b);
    }

    @Test
    void validatesInputTokenIdsAndContextLength() {
        GptModel model = new GptModel(4, 2, 4, 2, 1, 8, 1L);
        assertThrows(IllegalArgumentException.class, () -> model.forward(new int[]{}));
        assertThrows(IllegalArgumentException.class, () -> model.forward(new int[]{1, 2, 3}));
        assertThrows(IllegalArgumentException.class, () -> model.forward(new int[]{4}));
        assertThrows(IllegalArgumentException.class, () -> model.forward(new int[]{-1}));
        assertThrows(NullPointerException.class, () -> model.forward(null));
    }

    @Test
    void rejectsInvalidConfigurations() {
        assertThrows(IllegalArgumentException.class, () -> new GptModel(0, 8, 4, 2, 1, 8, 1L));
        assertThrows(IllegalArgumentException.class, () -> new GptModel(5, 0, 4, 2, 1, 8, 1L));
        assertThrows(IllegalArgumentException.class, () -> new GptModel(5, 8, 4, 3, 1, 8, 1L));
        assertThrows(IllegalArgumentException.class, () -> new GptModel(5, 8, 4, 2, 0, 8, 1L));
    }
}
