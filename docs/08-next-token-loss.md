# Lesson 8: Next-token cross-entropy loss

Before a language model can learn, it needs a way to **measure prediction error**.

## Shifted targets

For a sequence of token IDs `[A, B, C, D]`, the model produces a row of vocabulary scores (logits) at each position:

| Logit row | Correct next token |
| --- | --- |
| A | B |
| B | C |
| C | D |
| D | No target in this sequence |

We calculate loss over the first three rows and average. The last row has no known next token in this example, so it is excluded.

## Cross-entropy

For a target token `t` and vocabulary logits `z`, cross-entropy is:

```text
loss = log(sum(exp(z[j]))) - z[t]
```

The implementation subtracts the maximum logit before exponentiating, avoiding numerical overflow. Lower loss means higher probability assigned to the correct next token.

## Run

```bash
mvn verify
mvn exec:java -Dexec.mainClass=com.javiergs.llm.training.NextTokenLossExample
```

## Important distinction

This lesson **only measures loss**. It does not compute gradients, update weights, or train the model. ND4J supplies tensor operations, but our manually implemented forward pass does not automatically provide derivatives.

## Next: choose a gradient strategy

Before building a full training loop, we need a deliberate approach to differentiation:

- **Manual backpropagation** through each layer: transparent for teaching, but substantial work.
- **An automatic-differentiation system**: less derivative code, but requires integrating and potentially adapting the current forward-pass implementation.

A later PR should select and document one approach before adding an optimizer and training loop.
