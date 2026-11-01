# Lesson 12: Feed-forward network gradients

Each transformer block contains a position-wise feed-forward network. It applies two linear projections separated by **GELU**:

```text
z = x × W1
a = GELU(z)
y = a × W2
```

Our forward implementation uses the **tanh approximation** of GELU, so the backward pass differentiates that same approximation.

## Backward pass

Given upstream gradient `dy`, the chain rule yields:

```text
dW2 = transpose(a) × dy
da  = dy × transpose(W2)
dz  = da * GELU'(z)        (* means element-wise multiplication)
dW1 = transpose(x) × dz
dx  = dz × transpose(W1)
```

For `k = sqrt(2/pi)` and `t = tanh(k * (z + 0.044715 * z^3))`:

```text
GELU'(z) = 0.5 * (1 + t)
         + 0.5 * z * (1 - t^2) * k * (1 + 3 * 0.044715 * z^2)
```

The feed-forward operation processes each sequence position independently. The weight gradients sum contributions across all positions.

## Verify

Run `mvn verify`. JUnit compares gradients for the input, first projection, and second projection against central finite differences.

## Scope

The `FeedForwardGradients` class demonstrates the local backward pass using explicit weight matrices. The existing `FeedForward` class and model weights remain unchanged; parameter updates and end-to-end training will be added in later lessons.
