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

## Why are the example's predictions untrained?

The example initializes random weights, so its logits are not meaningful language predictions. For a trained model and generated characters, follow [Lesson 21](21-train-and-generate.md).

## Limitations

This educational model processes one sequence at a time, uses character-level tokens, and does not support dropout, batching, or saved checkpoints. A training loop is available in [Lesson 19](19-training-loop.md). The vocabulary projection has independent weights rather than being tied to the token embedding table.

## Exercise

Change the number of transformer blocks from two to three. Why does the output still have `V` columns? What happens when the prompt exceeds the context length?
