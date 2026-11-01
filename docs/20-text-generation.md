# Lesson 20: Autoregressive text generation

A decoder-only language model predicts the next token from the tokens already seen. **Autoregressive generation** repeats this prediction, appending each chosen token to the prompt.

This lesson uses **greedy decoding**: select the token with the highest logit. It is deterministic, making examples easy to reproduce and debug.

## Example

```java
CharacterTokenizer tokenizer = new CharacterTokenizer("abababab");
GptModel model = new GptModel(tokenizer.vocabularySize(), 4, 4, 2, 1, 6, 42);
String generated = TextGenerator.generateText(model, tokenizer, "ab", 8, 4);
System.out.println(generated);
```

Imports: `com.javiergs.llm.tokenizer.CharacterTokenizer`, `com.javiergs.llm.model.GptModel`, and `com.javiergs.llm.generation.TextGenerator`.

The model above is **randomly initialized**, so its output will not be meaningful. To generate learned patterns, train the model first using `GptTrainer` and token sequences from the same tokenizer.

## Generation algorithm

1. Encode the prompt as token IDs.
2. Keep the most recent tokens that fit the model's context window.
3. Run the model forward and read the **last row** of logits.
4. Choose the vocabulary token with the highest score.
5. Append it and repeat until the requested number of new tokens is reached.
6. Decode the full token sequence back to characters.

When the sequence grows beyond the context window, the generator drops older tokens **from the next model input**, but preserves them in the returned output.

## Verify

Run `mvn verify`. Tests check greedy argmax behavior, long generation with context truncation, text decoding, and invalid arguments.

## Limitations

This is a minimal character-level demonstration. It does not implement temperature sampling, top-k sampling, stop tokens, or efficient attention caching. A small model trained on a tiny corpus may memorize local patterns; it is not a general-purpose LLM.
