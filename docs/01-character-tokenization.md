# Lesson 1: Character-level tokenization

A language model processes **token IDs**, not raw strings. Our first tokenizer assigns one integer ID to each distinct Unicode code point observed in a training corpus.

## Try it

```bash
mvn verify
mvn exec:java -Dexec.mainClass=com.javiergs.llm.tokenizer.TokenizationExample
```

The example constructs a vocabulary from `Hello LLM!`, converts the text into integers, then decodes those integers back to the original text.

## Design decisions

- **Deterministic IDs:** code points are sorted before IDs are assigned, so the same set of characters produces the same mapping.
- **Unicode:** Java code points are used instead of UTF-16 `char` values, so supplementary characters such as emoji remain intact.
- **Unknown characters:** encoding fails explicitly if a character was not present in the training corpus.
- **No ND4J yet:** tokenization is a discrete text operation. ND4J becomes relevant for embedding vectors and attention.

## Limitations

A character tokenizer is easy to understand but inefficient for natural language. Later lessons will implement byte-pair encoding (BPE). A production model must also save its vocabulary alongside the model weights.

## Exercise

Change the corpus and predict the vocabulary size and token IDs before running the example. What happens when you try to encode a character absent from the corpus?
