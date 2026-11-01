# Lesson 3: Positional embeddings

Token embeddings represent **which token** appears. A decoder-only transformer also needs information about **where** the token appears in a sequence.

We implement GPT-style **absolute positional embeddings**: a second matrix with one vector for each position in the context window.

```text
token IDs:                 [sequence length]
token embeddings:         [sequence length, dimension]
position embeddings:      [sequence length, dimension]
combined input = token embeddings + position embeddings
```

Unlike sinusoidal positional encodings, these vectors are randomly initialized parameters that will be learned during training.

## Run

```bash
mvn verify
mvn exec:java -Dexec.mainClass=com.javiergs.llm.embedding.PositionalEmbeddingExample
```

## Why addition?

Adding the two vectors preserves the embedding width while allowing the following attention layers to use both token identity and absolute position. Two occurrences of the same character start with the same token embedding, but have different combined vectors because their positions differ.

## Constraints

- The token and positional embedding dimensions must match.
- The sequence must fit within the configured context length.
- This lesson implements the forward pass only. Embedding weights are not trained yet.

## Exercise

Change the text to include repeated characters. Compare their token vectors and their combined vectors. Why are the combined vectors different?

## Next

Lesson 4 introduces scaled dot-product self-attention, which allows tokens to incorporate information from other positions.
