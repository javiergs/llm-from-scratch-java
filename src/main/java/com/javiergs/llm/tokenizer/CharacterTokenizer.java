package com.javiergs.llm.tokenizer;

import java.util.Arrays;
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.Objects;

/**
 * A character-level tokenizer with a vocabulary learned from a supplied corpus.
 * Each Unicode code point is a token; unknown characters are rejected.
 * IDs are assigned in ascending code-point order for reproducibility.
 *
 * @author Javier Gonzalez-Sanchez
 * @version 0.1.0 (2026-10-31)
 */
public final class CharacterTokenizer {
    private final int[] vocabulary;
    private final Map<Integer, Integer> tokenIds;

    /**
     * Builds a deterministic vocabulary from all code points in a corpus.
     *
     * @param corpus training text used to establish the vocabulary
     */
    public CharacterTokenizer(String corpus) {
        Objects.requireNonNull(corpus, "corpus");
        vocabulary = corpus.codePoints().distinct().sorted().toArray();
        tokenIds = new LinkedHashMap<>();
        for (int i = 0; i < vocabulary.length; i++) {
            tokenIds.put(vocabulary[i], i);
        }
    }

    /**
     * Encodes text as token IDs. All code points must be in the vocabulary.
     *
     * @param text input text
     * @return token IDs
     * @throws IllegalArgumentException if a code point is not in the vocabulary
     */
    public int[] encode(String text) {
        Objects.requireNonNull(text, "text");
        return text.codePoints().map(codePoint -> {
            Integer id = tokenIds.get(codePoint);
            if (id == null) {
                throw new IllegalArgumentException(
                        "Unknown code point: U+" + Integer.toHexString(codePoint).toUpperCase());
            }
            return id;
        }).toArray();
    }

    /**
     * Decodes token IDs to text.
     *
     * @param ids token IDs
     * @return decoded text
     * @throws IllegalArgumentException if any ID is out of range
     */
    public String decode(int[] ids) {
        Objects.requireNonNull(ids, "ids");
        StringBuilder result = new StringBuilder();
        for (int id : ids) {
            if (id < 0 || id >= vocabulary.length) {
                throw new IllegalArgumentException("Invalid token ID: " + id);
            }
            result.appendCodePoint(vocabulary[id]);
        }
        return result.toString();
    }

    /**
     * Returns the number of distinct code points in the vocabulary.
     *
     * @return vocabulary size
     */
    public int vocabularySize() {
        return vocabulary.length;
    }

    /**
     * Returns a defensive copy of the sorted vocabulary code points.
     *
     * @return Unicode code points in token-ID order
     */
    public int[] vocabulary() {
        return Arrays.copyOf(vocabulary, vocabulary.length);
    }
}
