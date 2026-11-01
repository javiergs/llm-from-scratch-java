# Lesson 6: GPT-style transformer block

A transformer block combines two operations: **multi-head causal self-attention** (tokens communicate with earlier positions) and a **position-wise feed-forward network** (each token transforms its features).

## Pre-normalization and residual connections

This lesson uses the pre-norm architecture:

```text
x = token embeddings + positional embeddings

a = x + MultiHeadAttention(LayerNorm(x))
y = a + FeedForward(LayerNorm(a))
```

The residual additions preserve the model dimension and give each sublayer a direct path to its input. Layer normalization normalizes each token independently across its features, not across time positions. This preserves causal behavior.

The feed-forward network expands features from `d_model` to `d_ff`, applies the approximate GELU activation, then projects back to `d_model`. It never mixes tokens across sequence positions.

## Run

```bash
mvn verify
mvn exec:java -Dexec.mainClass=com.javiergs.llm.transformer.TransformerBlockExample
```

## Limitations

- Parameters are randomly initialized in this standalone example. See [Lesson 19](19-training-loop.md) for training the assembled model.
- Layer normalization uses its initial affine parameters (scale 1, shift 0) without trainable parameter updates.
- No dropout, batching, or optimizer is implemented.
- The implementation prioritizes clarity over vectorized performance.

## Exercise

Why would normalizing over all sequence positions break causal independence? Change the hidden dimension from 16 to 8 and verify that the block's output width remains 4.

## Next

Stack multiple blocks and add a vocabulary projection to assemble the forward pass of a small GPT-style model.
