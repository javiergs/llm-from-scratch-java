# Lesson 7: Assemble a small GPT-style model

This lesson connects all the previous components into an **untrained decoder-only transformer**.

```text
text → character tokenizer → token IDs
                              ↓
                token + position embeddings
                              ↓
                   transformer block 1
                              ↓
                   transformer block 2
                              ↓
                    final layer norm
                              ↓
                 vocabulary projection
                              ↓
                  logits at each position
```

## Run

```bash
mvn verify
mvn exec:java -Dexec.mainClass=com.javiergs.llm.model.GptModelExample
```

The example uses two transformer blocks, two attention heads, and an embedding width of eight. All weights are randomly initialized.

## Understanding logits

For a sequence of length `T` and vocabulary size `V`, the model returns a matrix with shape `[T, V]`. Each row contains **unnormalized scores** for the next token after that position. These are not probabilities. A softmax can convert them into a distribution.

During training, each position's logits will be compared with the **following token** using cross-entropy loss. For example, input `[a,b,c]` predicts targets `[b,c,d]`.

## Why no generated text yet?

Without training, logits are arbitrary. A later lesson will add a loss function, gradient-based optimization, and then autoregressive generation.

## Limitations

This educational model currently handles one sequence at a time, uses character-level tokens, and does not support dropout, batching, saved checkpoints, or a training loop. The vocabulary projection has independent weights rather than being tied to the token embedding table.

## Exercise

Change the number of transformer blocks from two to three. Why does the output still have `V` columns? What happens when the prompt exceeds the context length?
