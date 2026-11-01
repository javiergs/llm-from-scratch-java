package com.javiergs.llm.generation;

import com.javiergs.llm.model.GptModel;
import com.javiergs.llm.tokenizer.CharacterTokenizer;
import java.util.Arrays;
import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

/**
 * Tests greedy decoding, context-window truncation, and text round trips.
 *
 * @author Javier Gonzalez-Sanchez
 * @version 0.1.0 (2026-10-31)
 */
class TextGeneratorTest {
    @Test
    void greedyGenerationMatchesModelLastLogitArgmax() {
        GptModel model = new GptModel(3, 3, 4, 2, 1, 6, 42);
        int[] prompt = {0, 1};
        int[] generated = TextGenerator.generate(model, prompt, 1, 3);
        int expected = 0;
        double max = Double.NEGATIVE_INFINITY;
        var logits = model.forward(prompt);
        for (int token = 0; token < 3; token++) {
            if (logits.getDouble(1, token) > max) {
                max = logits.getDouble(1, token);
                expected = token;
            }
        }
        assertArrayEquals(new int[]{0, 1, expected}, generated);
        assertArrayEquals(new int[]{0, 1}, prompt);
    }

    @Test
    void longGenerationUsesSlidingContextWindow() {
        GptModel model = new GptModel(3, 3, 4, 2, 1, 6, 42);
        int[] output = TextGenerator.generate(model, new int[]{0, 1, 2}, 7, 3);
        assertEquals(10, output.length);
        for (int token : output) {
            assertTrue(token >= 0 && token < 3);
        }
        assertArrayEquals(new int[]{0, 1, 2}, Arrays.copyOf(output, 3));
    }

    @Test
    void decodesGeneratedCharactersAndChecksVocabulary() {
        CharacterTokenizer tokenizer = new CharacterTokenizer("ababa");
        GptModel model = new GptModel(2, 3, 4, 2, 1, 6, 42);
        String result = TextGenerator.generateText(model, tokenizer, "ab", 4, 3);
        assertTrue(result.startsWith("ab"));
        assertEquals(6, result.codePointCount(0, result.length()));
        assertThrows(IllegalArgumentException.class,
                () -> TextGenerator.generateText(model, tokenizer, "z", 1, 3));
    }

    @Test
    void rejectsEmptyPromptAndInvalidArguments() {
        GptModel model = new GptModel(2, 3, 4, 2, 1, 6, 42);
        assertThrows(IllegalArgumentException.class,
                () -> TextGenerator.generate(model, new int[]{}, 1, 3));
        assertThrows(IllegalArgumentException.class,
                () -> TextGenerator.generate(model, new int[]{0}, -1, 3));
        assertThrows(IllegalArgumentException.class,
                () -> TextGenerator.generate(model, new int[]{2}, 0, 3));
    }
}
