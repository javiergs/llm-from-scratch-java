# LLM from Scratch in Java

![Java 21+](https://img.shields.io/badge/Java-21%2B-blue)
![Version 0.1.0](https://img.shields.io/badge/version-0.1.0-green)
[![Java CI](https://github.com/javiergs/llm-from-scratch-java/actions/workflows/ci.yml/badge.svg)](https://github.com/javiergs/llm-from-scratch-java/actions/workflows/ci.yml)

An educational GPT-style language model built from scratch with **Java 21**, **Maven**, and **ND4J**.

ND4J handles numerical tensor operations; the model algorithms are implemented in Java rather than delegated to a pretrained model API.

## Learning path

The repository builds a small **decoder-only GPT-style language model** in incremental lessons. Start with the forward pass, then learn how gradients make training possible.

| Lesson | Topic | Status |
| --- | --- | --- |
| [1](docs/01-character-tokenization.md) | Character tokenization | Implemented |
| [2](docs/02-token-embeddings.md) | Token embeddings | Implemented |
| [3](docs/03-positional-embeddings.md) | Positional embeddings | Implemented |
| [4](docs/04-causal-self-attention.md) | Causal self-attention | Implemented |
| [5](docs/05-multi-head-attention.md) | Multi-head attention | Implemented |
| [6](docs/06-transformer-block.md) | Transformer block | Implemented |
| [7](docs/07-gpt-model.md) | GPT forward pass | Implemented |
| [8](docs/08-next-token-loss.md) | Next-token cross-entropy | Implemented |
| [9](docs/09-output-projection-gradients.md) | Output projection gradients | Implemented |

**In review:** [Lesson 10 — embedding gradients](https://github.com/javiergs/llm-from-scratch-java/pull/14) covers gradient accumulation for token and positional embedding tables.

**Coming next:** backward passes for layer normalization, feed-forward networks, attention, and complete transformer blocks; an optimizer and training loop; text generation; and optionally byte-pair encoding (BPE).

> **Current scope:** The model performs a complete forward pass and computes loss. Lessons 9–10 introduce standalone analytical gradients, but they are **not yet connected into end-to-end training**. Weights remain randomly initialized; this project does not yet train a language model.

## Run

Requires **Java 21+** and **Maven 3.9+**.

```bash
mvn verify
mvn exec:java -Dexec.mainClass=com.javiergs.llm.MatrixExample
```

The first command runs the JUnit test suite. The second runs the introductory ND4J matrix example. Follow the linked lessons for the model components and their examples.

