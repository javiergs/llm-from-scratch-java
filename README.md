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

Character tokenization → BPE → embeddings → self-attention → multi-head attention → transformer blocks → GPT → training and text generation.

Training will require an explicit gradient strategy beyond ND4J tensor operations.
