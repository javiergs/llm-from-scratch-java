# Lesson 2: Token embeddings with ND4J

The character tokenizer converts text to integer IDs. A neural language model needs **vectors**, not just IDs. A token embedding is a lookup table represented as a matrix:

```text
weights: [vocabulary size, embedding dimension]
input:   [sequence length] token IDs
output:  [sequence length, embedding dimension]
```

Each token ID selects a row of the weight matrix. Repeated token IDs select identical vectors.

## Run

```bash
mvn verify
mvn exec:java -Dexec.mainClass=com.javiergs.llm.embedding.EmbeddingExample
```

The example uses a four-dimensional vector for each character in `Hello LLM!`. This tiny dimension is for illustration, not language-model quality.

## What ND4J does

ND4J stores embedding weights as an `INDArray`. This lesson explicitly performs the lookup rather than using a pretrained embedding model. The example uses a fixed seed and small random initial values for reproducible results.

## Important limitation

These are **randomly initialized, untrained embeddings**. They do not encode learned semantic similarity until trained. [Lessons 18–19](18-end-to-end-model-gradients.md) connect gradients and optimization to the complete model; this lesson focuses on the forward lookup.

## Exercise

Try changing the embedding dimension from 4 to 8. Predict the output shape before running the example. Why do repeated characters have identical vectors?
