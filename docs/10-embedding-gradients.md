# Lesson 10: Embedding gradients

Token embeddings are a **lookup table**: each token ID selects one row of a weight matrix. Positional embeddings select a row based on the token's position. Their forward outputs are added together.

## Backward pass

For the combined vector at position `i`:

```text
combined[i] = tokenWeights[tokenIds[i]] + positionWeights[i]
```

The gradient of the combined vector is passed to **both** inputs of this addition.

- **Token table:** Add each position's upstream gradient to the row corresponding to its token ID. If the same token appears several times, **sum** those contributions.
- **Position table:** Copy each position's upstream gradient into that position's row. Rows outside the sequence receive zero.

These are *parameter gradients*, not updates. We still need to propagate gradients through the intervening transformer layers before they can be connected to the output loss.

## Tests

Run `mvn verify`. The tests cover repeated token IDs, unused rows, invalid shapes and IDs, and a central finite-difference check for token lookup.

## Scope

`EmbeddingGradients` is intentionally independent of `GptModel`. It demonstrates the local backward rule for lookup and addition without exposing or modifying the model's private weights. A later lesson will connect these components into a complete backward pass.
