# Lesson 21: Train and generate with a tiny corpus

This example puts the lessons together in a single runnable Java program. It uses a deliberately tiny repeated-character corpus so the model can learn a recognizable pattern without downloading data or using pretrained weights.

## Run

Requires Java 21 and Maven:

```bash
mvn verify
mvn exec:java -Dexec.mainClass=com.javiergs.llm.training.TrainAndGenerateExample
```

## What happens

1. Construct a character-level vocabulary from `abababababababab`.
2. Encode the text into integer token IDs.
3. Initialize a small decoder-only GPT model with a reproducible random seed.
4. Split the corpus into short sequences containing next-token targets.
5. Measure initial mean next-token cross-entropy.
6. Train using the complete backward pass and SGD for 12 epochs.
7. Print training loss after each epoch.
8. Generate 12 additional characters from the prompt `ab` using greedy decoding.

The program is intentionally small and self-contained. It trains on a repeated pattern, not natural language; any generated pattern is evidence of learning a tiny training distribution, **not** general language competence.

## Explore

Change the corpus, number of epochs, learning rate, or prompt. Every prompt character must exist in the training vocabulary. Large learning rates may make loss increase; training loss is not a held-out evaluation metric.

## Next steps

Potential extensions include preparing longer corpora, validation splits, mini-batches, checkpointing, sampling strategies, and faster attention. The core educational training-and-generation path is now complete.
