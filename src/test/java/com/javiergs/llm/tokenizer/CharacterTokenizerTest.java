package com.javiergs.llm.tokenizer;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Tests round-trip encoding, deterministic IDs, Unicode, and invalid input.
 *
 * @author Javier Gonzalez-Sanchez
 * @version 0.1.0 (2026-10-31)
 */
class CharacterTokenizerTest {

    @Test
    void roundTripPreservesText() {
        CharacterTokenizer tokenizer = new CharacterTokenizer("Hello LLM!");
        assertEquals("Hello LLM!", tokenizer.decode(tokenizer.encode("Hello LLM!")));
    }

    @Test
    void vocabularyIsDeterministic() {
        CharacterTokenizer first = new CharacterTokenizer("cbaac");
        CharacterTokenizer second = new CharacterTokenizer("abc");
        assertArrayEquals(first.vocabulary(), second.vocabulary());
        assertArrayEquals(new int[]{0, 1, 2}, first.encode("abc"));
        assertEquals(3, first.vocabularySize());
    }

    @Test
    void handlesSupplementaryUnicodeCodePoints() {
        CharacterTokenizer tokenizer = new CharacterTokenizer("A😀é");
        assertEquals(3, tokenizer.vocabularySize());
        assertEquals("😀Aé", tokenizer.decode(tokenizer.encode("😀Aé")));
    }

    @Test
    void rejectsUnknownCharacters() {
        CharacterTokenizer tokenizer = new CharacterTokenizer("abc");
        assertThrows(IllegalArgumentException.class, () -> tokenizer.encode("abd"));
    }

    @Test
    void rejectsInvalidTokenIds() {
        CharacterTokenizer tokenizer = new CharacterTokenizer("abc");
        assertThrows(IllegalArgumentException.class, () -> tokenizer.decode(new int[]{3}));
        assertThrows(IllegalArgumentException.class, () -> tokenizer.decode(new int[]{-1}));
    }

    @Test
    void supportsEmptyCorpusAndEmptyText() {
        CharacterTokenizer tokenizer = new CharacterTokenizer("");
        assertEquals(0, tokenizer.vocabularySize());
        assertArrayEquals(new int[0], tokenizer.encode(""));
        assertEquals("", tokenizer.decode(new int[0]));
    }
}
