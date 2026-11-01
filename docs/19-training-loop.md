# Lesson 19: Training a GPT-style model

The complete learning cycle consists of a forward pass, next-token loss, analytical gradients, and SGD parameter updates.

A **training epoch** processes every sequence once. Each sequence produces one SGD update. At the end of an epoch, the trainer evaluates mean cross-entropy on the supplied training examples.

## Example

```java
GptModel model = new GptModel(3, 4, 4, 2, 1, 6, 42);
List<int[]> corpus = List.of(new int[]{0, 1, 0, 1});

double before = GptTrainer.meanLoss(model, corpus);
GptTrainer.TrainingHistory history = new GptTrainer(0.01).train(model, corpus, 8);
double after = GptTrainer.meanLoss(model, corpus);

System.out.println("Before: " + before);
System.out.println("After: " + after);
System.out.println("Loss by epoch: " + history.epochLosses());
```

Imports: `com.javiergs.llm.model.GptModel`, `com.javiergs.llm.training.GptTrainer`, and `java.util.List`.

The example uses integer token IDs, not a text dataset. [Lesson 21](21-train-and-generate.md) connects tokenization, training examples, and text generation.

## How the loop works

1. Validate each sequence and its token IDs.
2. Compute next-token gradients through all model layers.
3. Apply SGD to all trainable parameters.
4. Repeat for the requested number of epochs.
5. Measure mean training loss after each epoch.

Loss is weighted by the number of predicted next tokens, so longer examples contribute proportionally more to the reported average.

## Verify

Run `mvn verify`. Tests check that training reduces loss on a tiny repeated-token pattern, that evaluation leaves weights unchanged, and that invalid examples are rejected.

## Limitations

This is an intentionally small educational trainer. It has no batching, held-out evaluation, checkpointing, learning-rate schedule, gradient clipping, or optimizer state. Training loss is **not** a measure of generalization, and loss is not guaranteed to decrease on every epoch for arbitrary data or learning rates.
