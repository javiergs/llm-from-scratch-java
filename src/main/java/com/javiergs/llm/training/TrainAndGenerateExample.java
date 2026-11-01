package com.javiergs.llm.training;

import com.javiergs.llm.generation.TextGenerator;
import com.javiergs.llm.model.GptModel;
import com.javiergs.llm.tokenizer.CharacterTokenizer;
import java.util.List;

/**
 * Demonstrates the complete educational workflow: tokenize a tiny corpus,
 * train a GPT-style model, and generate characters from a prompt.
 *
 * @author Javier Gonzalez-Sanchez
 * @version 0.1.0 (2026-10-31)
 */
public final class TrainAndGenerateExample {
    private TrainAndGenerateExample() {
    }

    /**
     * Trains on a repeated character pattern and prints generated text.
     *
     * @param args unused command-line arguments
     */
    public static void main(String[] args) {
        String corpus = "abababababababab";
        CharacterTokenizer tokenizer = new CharacterTokenizer(corpus);
        int[] tokens = tokenizer.encode(corpus);

        int contextLength = 4;
        GptModel model = new GptModel(
                tokenizer.vocabularySize(), contextLength, 4, 2, 1, 8, 42L);

        // Each window includes its next-token targets. Adjacent windows overlap.
        List<int[]> examples = new java.util.ArrayList<>();
        for (int start = 0; start + contextLength <= tokens.length; start += contextLength) {
            examples.add(java.util.Arrays.copyOfRange(tokens, start, start + contextLength));
        }

        System.out.println("Corpus: " + corpus);
        System.out.println("Vocabulary size: " + tokenizer.vocabularySize());
        System.out.printf("Initial training loss: %.4f%n", GptTrainer.meanLoss(model, examples));

        GptTrainer.TrainingHistory history = new GptTrainer(0.01).train(model, examples, 12);
        for (int epoch = 0; epoch < history.epochLosses().size(); epoch++) {
            System.out.printf("Epoch %2d | training loss %.4f%n",
                    epoch + 1, history.epochLosses().get(epoch));
        }

        String prompt = "ab";
        String generated = TextGenerator.generateText(
                model, tokenizer, prompt, 12, contextLength);
        System.out.println("Prompt: " + prompt);
        System.out.println("Generated: " + generated);
        System.out.println("This toy corpus illustrates memorization, not general language understanding.");
    }
}
