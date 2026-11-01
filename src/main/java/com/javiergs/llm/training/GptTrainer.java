package com.javiergs.llm.training;

import com.javiergs.llm.model.GptModel;
import java.util.ArrayList;
import java.util.List;
import org.nd4j.linalg.api.ndarray.INDArray;

/**
 * Trains a small GPT model on sequences of integer token IDs using SGD.
 * Each sequence supplies its own next-token targets; sequences are processed
 * in the supplied order, one optimizer update at a time.
 *
 * @author Javier Gonzalez-Sanchez
 * @version 0.1.0 (2026-10-31)
 */
public final class GptTrainer {
    private final StochasticGradientDescent optimizer;

    /**
     * Configures a trainer with a constant SGD learning rate.
     *
     * @param learningRate finite positive optimizer step size
     */
    public GptTrainer(double learningRate) {
        optimizer = new StochasticGradientDescent(learningRate);
    }

    /**
     * Stores mean next-token cross-entropy measured after each epoch.
     *
     * @param epochLosses one mean loss per completed epoch
     */
    public record TrainingHistory(List<Double> epochLosses) {
        /**
         * Protects the history from external modification.
         *
         * @param epochLosses completed epoch losses
         */
        public TrainingHistory {
            epochLosses = List.copyOf(epochLosses);
        }
    }

    /**
     * Performs one SGD update per sequence and measures mean loss after each
     * epoch. The measured loss is an evaluation on the training sequences.
     *
     * @param model model to update
     * @param sequences nonempty training examples with at least two tokens each
     * @param epochs positive number of passes through the training examples
     * @return mean training loss measured at the end of every epoch
     */
    public TrainingHistory train(GptModel model, List<int[]> sequences, int epochs) {
        validate(model, sequences);
        if (epochs <= 0) {
            throw new IllegalArgumentException("Epoch count must be positive");
        }
        List<Double> losses = new ArrayList<>();
        for (int epoch = 0; epoch < epochs; epoch++) {
            for (int[] sequence : sequences) {
                INDArray[] gradients = GptModelGradients.backward(model, sequence);
                optimizer.step(model, gradients);
            }
            losses.add(meanLoss(model, sequences));
        }
        return new TrainingHistory(losses);
    }

    /**
     * Evaluates mean next-token cross-entropy across supplied sequences.
     * No model parameters are modified.
     *
     * @param model model to evaluate
     * @param sequences nonempty sequences with at least two tokens each
     * @return mean loss weighted by the number of next-token predictions
     */
    public static double meanLoss(GptModel model, List<int[]> sequences) {
        validate(model, sequences);
        double total = 0;
        long predictions = 0;
        for (int[] sequence : sequences) {
            long count = sequence.length - 1L;
            total += GptModelGradients.loss(model, sequence) * count;
            predictions += count;
        }
        return total / predictions;
    }

    private static void validate(GptModel model, List<int[]> sequences) {
        if (model == null || sequences == null) {
            throw new NullPointerException("Model and sequences must not be null");
        }
        if (sequences.isEmpty()) {
            throw new IllegalArgumentException("At least one sequence is required");
        }
        for (int[] sequence : sequences) {
            if (sequence == null || sequence.length < 2) {
                throw new IllegalArgumentException("Each sequence must contain at least two tokens");
            }
            // Validates token IDs and maximum context length before training begins.
            model.forward(sequence);
        }
    }
}
