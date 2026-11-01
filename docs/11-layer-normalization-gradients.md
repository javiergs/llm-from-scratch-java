# Lesson 11: Layer normalization gradients

Layer normalization stabilizes transformer activations by normalizing **each token's feature vector**. The forward implementation in `LayerNormalization` uses the population mean and variance, with a small positive `epsilon`.

## Forward

For a row with `D` features:

```text
mean = sum(x) / D
variance = sum((x - mean)^2) / D
normalized = (x - mean) / sqrt(variance + epsilon)
```

## Backward

Let `g` be the gradient arriving from the next layer and `y` the normalized row. Then:

```text
dx = (g - mean(g) - y * mean(g * y)) / sqrt(variance + epsilon)
```

The means in this expression are taken **across the features of one row**, not across the entire sequence. Both the mean and variance depend on the input, so simply dividing the upstream gradient by the standard deviation would be incorrect.

This project currently uses layer normalization **without learned scale or shift parameters**, so this lesson only computes input gradients.

## Verify

Run `mvn verify`. The tests compare the analytical derivative against central finite differences of the existing forward implementation, and cover constant rows and one-feature vectors.

## Scope

This is a standalone backward rule. It does not change `LayerNormalization` or update model parameters. Later lessons will connect it to the transformer block's complete backward pass.
