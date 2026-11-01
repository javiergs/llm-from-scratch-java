# LLM from Scratch in Java

![Java 21+](https://img.shields.io/badge/Java-21%2B-blue)
![Version 0.1.0](https://img.shields.io/badge/version-0.1.0-green)
[![Java CI](https://github.com/javiergs/llm-from-scratch-java/actions/workflows/ci.yml/badge.svg)](https://github.com/javiergs/llm-from-scratch-java/actions/workflows/ci.yml)

An educational GPT-style language model built from scratch with **Java 21**, **Maven**, and **ND4J**.

ND4J handles numerical tensor operations; the model algorithms are implemented in Java rather than delegated to a pretrained model API.

## Get started

**Recommended preparation:** Basic Java programming, arrays, matrix multiplication, and familiarity with Maven. Knowledge of neural networks is helpful but not required.

Requires **Java 21+** and **Maven 3.9+**. Clone the repository, then run the tests:

```bash
git clone https://github.com/javiergs/llm-from-scratch-java.git
cd llm-from-scratch-java
```

Run the JUnit tests:

```bash
mvn verify
```

For a first look at ND4J matrix operations, run:

```bash
mvn exec:java -Dexec.mainClass=com.javiergs.llm.MatrixExample
```

For the **complete training and generation demonstration**, run:

```bash
mvn exec:java -Dexec.mainClass=com.javiergs.llm.training.TrainAndGenerateExample
```

This example builds a vocabulary from a tiny repeated-character corpus, trains a small GPT-style model, prints training loss after each epoch, and generates characters from a prompt. See [Lesson 21](docs/21-train-and-generate.md) for the walkthrough.

## Learning path

**New to the project?** Start with [Lesson 1](docs/01-character-tokenization.md) and follow the lessons in order. Each lesson introduces a specific concept; the final lesson demonstrates the complete workflow.

The repository builds a small **decoder-only GPT-style language model** in incremental lessons. Start with the forward pass, then learn how gradients make training possible.

1. [Character tokenization](docs/01-character-tokenization.md)
2. [Token embeddings](docs/02-token-embeddings.md)
3. [Positional embeddings](docs/03-positional-embeddings.md)
4. [Causal self-attention](docs/04-causal-self-attention.md)
5. [Multi-head attention](docs/05-multi-head-attention.md)
6. [Transformer block](docs/06-transformer-block.md)
7. [GPT forward pass](docs/07-gpt-model.md)
8. [Next-token cross-entropy](docs/08-next-token-loss.md)
9. [Output projection gradients](docs/09-output-projection-gradients.md)
10. [Embedding gradients](docs/10-embedding-gradients.md)
11. [Layer normalization gradients](docs/11-layer-normalization-gradients.md)
12. [Feed-forward network gradients](docs/12-feed-forward-gradients.md)
13. [Causal self-attention gradients](docs/13-causal-attention-gradients.md)
14. [Multi-head attention gradients](docs/14-multi-head-attention-gradients.md)
15. [Transformer block gradients](docs/15-transformer-block-gradients.md)
16. [Trainable model parameters](docs/16-trainable-model-parameters.md)
17. [Stochastic gradient descent](docs/17-stochastic-gradient-descent.md)
18. [End-to-end GPT gradients](docs/18-end-to-end-model-gradients.md)
19. [Training loop](docs/19-training-loop.md)
20. [Autoregressive text generation](docs/20-text-generation.md)
21. [Complete training and generation example](docs/21-train-and-generate.md)

**Current scope:** This repository implements a complete educational character-level GPT training and generation pipeline. It uses small datasets and a simple SGD optimizer; it is not a pretrained or production-scale language model.

## What this project does not do

- It does **not** download or use pretrained model weights.
- It does **not** train on a large natural-language corpus or demonstrate general language understanding.
- It does **not** include production features such as checkpointing, batching, or efficient cached inference.

The goal is to make the essential algorithms inspectable and understandable in Java, one lesson at a time.
