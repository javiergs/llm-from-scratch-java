# LLM from Scratch in Java

![Java 21+](https://img.shields.io/badge/Java-21%2B-blue)
![Version 0.1.0](https://img.shields.io/badge/version-0.1.0-green)
[![Java CI](https://github.com/javiergs/llm-from-scratch-java/actions/workflows/ci.yml/badge.svg)](https://github.com/javiergs/llm-from-scratch-java/actions/workflows/ci.yml)

An educational GPT-style language model built from scratch with **Java 21**, **Maven**, and **ND4J**.

ND4J handles numerical tensor operations; the model algorithms are implemented in Java rather than delegated to a pretrained model API.

## Learning path

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

**In review:** [Lesson 12 — feed-forward network gradients](https://github.com/javiergs/llm-from-scratch-java/pull/17).

**Coming next:** backward passes for attention and complete transformer blocks; an optimizer and training loop; text generation; and optionally byte-pair encoding (BPE).

> **Current scope:** The model performs a complete forward pass and computes loss. Lessons 9–12 introduce standalone analytical gradients, but they are **not yet connected into end-to-end training**. Weights remain randomly initialized; this project does not yet train a language model.

## Run

Requires **Java 21+** and **Maven 3.9+**.

```bash
mvn verify
mvn exec:java -Dexec.mainClass=com.javiergs.llm.MatrixExample
```

The first command runs the JUnit test suite. The second runs the introductory ND4J matrix example. Follow the linked lessons for the model components and their examples.

