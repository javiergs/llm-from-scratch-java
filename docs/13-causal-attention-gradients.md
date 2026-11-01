# Lesson 13: Causal self-attention gradients

Causal self-attention lets each token use information from its own position and earlier positions, but **never from future positions**.

## Forward pass

```text
Q = X × Wq
K = X × Wk
V = X × Wv
S = Q × transpose(K) / sqrt(D)
P = causalSoftmax(S)
Y = P × V
```

For each row `i`, causal softmax only includes columns `j <= i`. All future-position probabilities are zero.

## Backward pass

Starting with `dY`:

```text
dV = transpose(P) × dY
dP = dY × transpose(V)
```

For each row, define `weighted = sum_j(P[i,j] * dP[i,j])` over **unmasked** positions. Then:

```text
dS[i,j] = P[i,j] * (dP[i,j] - weighted)    when j <= i
dS[i,j] = 0                                when j > i
```

The remaining matrix derivatives are:

```text
dQ = dS × K / sqrt(D)
dK = transpose(dS) × Q / sqrt(D)
dWq = transpose(X) × dQ
dWk = transpose(X) × dK
dWv = transpose(X) × dV
dX = dQ × transpose(Wq) + dK × transpose(Wk) + dV × transpose(Wv)
```

## Verify

Run `mvn verify`. JUnit compares all four gradients (input, query weights, key weights, and value weights) against central finite differences. A causal-isolation test checks that a loss depending only on the first output cannot produce gradients at later input positions.

## Scope

This lesson covers **single-head attention**. Multi-head attention additionally slices Q/K/V into heads, concatenates their outputs, and applies an output projection. [Lesson 14](14-multi-head-attention-gradients.md) covers the multi-head backward pass. [Lesson 19](19-training-loop.md) demonstrates training the complete model.
