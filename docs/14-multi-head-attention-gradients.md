# Lesson 14: Multi-head attention gradients

Multi-head attention projects token vectors into queries, keys, and values, splits each projected vector into independent heads, performs **causal attention** per head, concatenates head outputs, and applies a final output projection.

## Forward

```text
Q = X × Wq, K = X × Wk, V = X × Wv
head[h] = causalSoftmax(Q[h] × transpose(K[h]) / sqrt(headDimension)) × V[h]
concatenated = concat(head[0], ..., head[H-1])
Y = concatenated × Wo
```

## Backward

First differentiate the final projection:

```text
dWo = transpose(concatenated) × dY
dConcatenated = dY × transpose(Wo)
```

Split `dConcatenated` into head-sized slices. For each head, apply the causal softmax and Q/K/V derivatives from [Lesson 13](13-causal-attention-gradients.md), scaling by `sqrt(headDimension)`, **not** the full model dimension.

Place each head's `dQ`, `dK`, and `dV` into the matching columns of the full projected gradients. Then:

```text
dWq = transpose(X) × dQ
dWk = transpose(X) × dK
dWv = transpose(X) × dV
dX = dQ × transpose(Wq) + dK × transpose(Wk) + dV × transpose(Wv)
```

Masked future positions contribute no gradients.

## Verify

Run `mvn verify`. The tests compare input and all four weight gradients against central finite differences using two heads.

## Scope

This is an educational standalone backward calculation. The model does not yet update weights or train end to end. The next lesson will connect the local backward rules within a transformer block.
