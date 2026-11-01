# Lesson 16: Accessing trainable model parameters

The earlier lessons computed gradients using explicit matrices. A training loop also needs a safe way to **read and replace the actual model weights**.

`GptModel.parameters()` returns **copies**, not references to the model's internal ND4J arrays. Modifying these copies cannot silently change the model. Call `GptModel.setParameters(...)` to replace weights after validating every matrix shape.

Parameter order:

1. Token embedding table.
2. Positional embedding table.
3. Six matrices for each transformer block: query, key, value, attention output, feed-forward input, feed-forward output.
4. Final vocabulary output projection.

The final layer normalization has no learned scale or bias in this educational implementation.

## Example

```java
INDArray[] weights = model.parameters();
INDArray[] updated = new INDArray[weights.length];
for (int i = 0; i < weights.length; i++) {
    updated[i] = weights[i].sub(gradients[i].mul(learningRate));
}
model.setParameters(updated);
```

This example illustrates how an optimizer **will** apply gradients; `gradients` and `learningRate` must be supplied by the future training lesson.

## Verify

Run `mvn verify`. Tests confirm defensive copies, explicit updates, and rejection of invalid shapes without partial updates.

## Scope

This lesson provides parameter access and replacement, **not** end-to-end gradient collection or training. [Lesson 18](18-end-to-end-model-gradients.md) connects backward calculations to these parameters; [Lesson 17](17-stochastic-gradient-descent.md) introduces the optimizer.
