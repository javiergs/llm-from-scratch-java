# LLM from Scratch in Java

An educational GPT-style language model built from scratch with **Java 21**, **Maven**, and **ND4J**.

ND4J handles numerical tensor operations; the model algorithms are implemented in Java rather than delegated to a pretrained model API.

## Requirements

Java 21 and Maven 3.9+.

## Run

```bash
mvn verify
mvn exec:java -Dexec.mainClass=com.javiergs.llm.MatrixExample
```

The first example demonstrates ND4J matrix multiplication, verified with JUnit.

## Roadmap

The project builds a small GPT-style language model step by step:

1. **Character tokenization:** Convert text into integer token IDs. **BPE (planned)** will later group frequent character sequences into more efficient tokens.
2. **Embeddings:** Represent token identities and their positions as numerical vectors.
3. **Self-attention:** Let each token use information from earlier tokens, while masking future tokens.
4. **Multi-head attention:** Compute several attention patterns in parallel.
5. **Transformer blocks:** Combine attention, normalization, residual connections, and feed-forward networks.
6. **GPT model:** Stack transformer blocks and project their outputs into next-token vocabulary scores.
7. **Training (planned):** Measure prediction error and update model parameters using gradients.
8. **Text generation (planned):** Repeatedly predict and select the next token to extend a prompt.

**Training note:** ND4J provides tensor operations, but training also requires an explicit gradient-computation and optimization strategy. The current model implements the forward pass; its weights are not yet trained.
