# Lesson 9: Backpropagation through the output projection

The previous lesson measured **next-token cross-entropy**. Now we calculate how a small change to a parameter changes that loss.

## Step 1: Logits gradient

For each nonfinal position, cross-entropy has a convenient derivative:

```text
dLoss/dLogits[position, token]
    = (softmax(logits)[token] - isCorrectToken) / numberOfTargets
```

The last row has no target and therefore contributes a row of zeros. We use maximum-subtracted softmax for numerical stability.

## Step 2: Linear projection

The output layer computes:

```text
logits = hiddenStates × weights
```

Applying the chain rule gives:

```text
dLoss/dWeights      = transpose(hiddenStates) × dLoss/dLogits
dLoss/dHiddenStates = dLoss/dLogits × transpose(weights)
```

The weight gradient tells us how the vocabulary projection should change. The hidden-state gradient will eventually flow backward through normalization and transformer blocks.

## Verify the derivatives

Run `mvn verify`. JUnit compares analytical derivatives against **central finite differences**:

```text
derivative ≈ (loss(parameter + epsilon) - loss(parameter - epsilon)) / (2 × epsilon)
```

Finite differences are useful as tests, not as an efficient way to train a language model.

## Scope

This lesson isolates **output-projection gradient calculations** without modifying model parameters. [Lesson 18](18-end-to-end-model-gradients.md) connects these calculations to the complete model, and [Lesson 19](19-training-loop.md) trains it.
