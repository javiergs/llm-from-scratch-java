# Lesson 4: Causal self-attention

Attention lets a token use information from other tokens. In a GPT-style decoder, a token may only attend to **itself and earlier tokens**.

## Formula

```text
Q = X Wq,  K = X Wk,  V = X Wv
scores = Q Kᵀ / sqrt(d)
weights = causal_softmax(scores)
output = weights V
```

The causal mask sets all probabilities above the diagonal to zero. The softmax is computed only over permitted positions, using subtraction of the row maximum for numerical stability.

## Run

```bash
mvn verify
mvn exec:java -Dexec.mainClass=com.javiergs.llm.attention.AttentionExample
```

The example connects the preceding lessons: text → token IDs → token vectors + positional vectors → causal self-attention.

## What to inspect

- The probability matrix is lower triangular.
- Each row sums to approximately one.
- Changing a future token cannot change an earlier output.
- Q, K, and V are **different, randomly initialized projections**.
- These are untrained parameters; the output is not meaningful language understanding yet.

## Scope and next step

This lesson implements a **single head** with equal input, query/key, and value dimensions. Multi-head attention will split attention across several heads in the next lesson.

## Exercise

Replace the last character of the example text and compare attention outputs at earlier positions. Why must they remain unchanged?
