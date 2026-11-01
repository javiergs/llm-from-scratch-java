# Lesson 18: End-to-end GPT gradients

The complete backward pass connects the earlier gradient rules in this sequence:

1. Look up token and position embeddings.
2. Run each pre-normalization transformer block, retaining intermediate states.
3. Apply final layer normalization and the vocabulary projection.
4. Compute mean next-token cross-entropy.
5. Differentiate the output projection and final normalization.
6. Walk **backward** through the transformer blocks.
7. Accumulate gradients into token and positional embedding tables.

`GptModelGradients.backward(model, tokenIds)` returns a matrix for **every** trainable parameter, ordered exactly like `model.parameters()`.

## First genuine SGD step

```java
int[] sequence = {0, 1, 0, 1};
double before = GptModelGradients.loss(model, sequence);
INDArray[] gradients = GptModelGradients.backward(model, sequence);
new StochasticGradientDescent(0.01).step(model, gradients);
double after = GptModelGradients.loss(model, sequence);
System.out.println("Loss: " + before + " -> " + after);
```

This is the first lesson that connects next-token loss to **all** model parameters. [Lesson 19](19-training-loop.md) adds the reusable training loop, and [Lesson 21](21-train-and-generate.md) demonstrates training and text generation.

## Verify

Run `mvn verify`. Tests compare representative model gradients against finite differences and verify that a small SGD update decreases next-token cross-entropy.

## Scope

The implementation is deliberately educational: one sequence at a time, no batching, gradient clipping, checkpointing, or adaptive optimizer. Hidden-state snapshots are returned for backpropagation; they do not expose mutable internal parameter references.
