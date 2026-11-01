# Lesson 5: Multi-head causal attention

A single attention head learns one set of query, key, and value projections. Multi-head attention divides the model's projected features across several heads, computes causal attention independently, concatenates their results, and applies an output projection.

## Shapes

For model dimension `d_model = 4` and `heads = 2`:

```text
input:              [sequence length, 4]
Q, K, V:            [sequence length, 4]
each head Q/K/V:    [sequence length, 2]
each head output:   [sequence length, 2]
concatenated:       [sequence length, 4]
output projection:  [sequence length, 4]
```

The number of heads must divide the model dimension exactly. Each head uses scaling by `sqrt(d_head)`, not `sqrt(d_model)`.

## Run

```bash
mvn verify
mvn exec:java -Dexec.mainClass=com.javiergs.llm.attention.MultiHeadAttentionExample
```

The example shows each head's causal attention probability matrix. Both matrices are lower triangular, and each row sums to one.

## Educational notes

- All four projection matrices are initialized randomly and are **not trained yet**.
- This implementation uses ND4J for matrix multiplication and explicit Java loops for causal softmax, prioritizing readability over speed.
- Different heads can learn different relationships once training is implemented.
- No dropout, batching, or key/value caching is included in this lesson.

## Exercise

Try changing the number of heads from 2 to 1 or 4. What changes in each head's vector width? Why is a model dimension of 5 incompatible with 2 heads?

## Next

Transformer blocks combine multi-head attention with normalization, residual connections, and feed-forward layers.
