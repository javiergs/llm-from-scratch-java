package com.javiergs.llm.training;

import com.javiergs.llm.model.GptModel;
import java.util.List;
import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

/**
 * Tests reproducible small-corpus training and validation.
 *
 * @author Javier Gonzalez-Sanchez
 * @version 0.1.0 (2026-10-31)
 */
class GptTrainerTest {
    @Test
    void learnsSimpleRepeatedTokenPattern() {
        GptModel model = new GptModel(3, 4, 4, 2, 1, 6, 42);
        List<int[]> corpus = List.of(new int[]{0, 1, 0, 1});
        double before = GptTrainer.meanLoss(model, corpus);
        GptTrainer.TrainingHistory history = new GptTrainer(0.01).train(model, corpus, 8);
        assertEquals(8, history.epochLosses().size());
        double after = GptTrainer.meanLoss(model, corpus);
        assertTrue(Double.isFinite(after));
        assertTrue(after < before, "Training should improve loss on this small repeated pattern");
        assertEquals(after, history.epochLosses().get(7), 1e-6);
        assertThrows(UnsupportedOperationException.class, () -> history.epochLosses().add(1.0));
    }

    @Test
    void rejectsInvalidExamplesAndEpochCounts() {
        GptModel model = new GptModel(3, 4, 4, 2, 1, 6, 42);
        GptTrainer trainer = new GptTrainer(0.01);
        assertThrows(IllegalArgumentException.class, () -> trainer.train(model, List.of(), 1));
        assertThrows(IllegalArgumentException.class,
                () -> trainer.train(model, List.of(new int[]{0}), 1));
        assertThrows(IllegalArgumentException.class,
                () -> trainer.train(model, List.of(new int[]{0, 1}), 0));
        assertThrows(IllegalArgumentException.class,
                () -> trainer.train(model, List.of(new int[]{0, 5}), 1));
    }

    @Test
    void evaluationDoesNotChangeWeights() {
        GptModel model = new GptModel(3, 4, 4, 2, 1, 6, 42);
        var before = model.parameters();
        GptTrainer.meanLoss(model, List.of(new int[]{0, 1, 0}));
        var after = model.parameters();
        for (int i = 0; i < before.length; i++) {
            assertEquals(0, before[i].sub(after[i]).norm2Number().doubleValue(), 1e-6);
        }
    }
}
