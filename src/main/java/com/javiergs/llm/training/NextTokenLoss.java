package com.javiergs.llm.training;

import org.nd4j.linalg.api.ndarray.INDArray;

/**
 * Computes next-token cross-entropy from unnormalized vocabulary logits.
 * This lesson measures prediction error but does not update model parameters.
 *
 * @author Javier Gonzalez-Sanchez
 * @version 0.1.0 (2026-10-31)
 */
public final class NextTokenLoss {
    private NextTokenLoss() {
    }

    /**
     * Computes mean next-token cross-entropy using numerically stable log-sum-exp.
     * Logit row i predicts the token at position i + 1. The final row is
     * ignored because its next-token target is not present in the sequence.
     *
     * @param logits scores of shape [sequence length, vocabulary size]
     * @param tokenIds input token sequence, including the target tokens
     * @return average cross-entropy in natural logarithm units
     */
    public static double mean(INDArray logits, int[] tokenIds) {
        if (logits == null || tokenIds == null) {
            throw new NullPointerException("logits and tokenIds must not be null");
        }
        if (logits.rank() != 2 || logits.size(0) != tokenIds.length
                || tokenIds.length < 2 || logits.size(1) <= 0) {
            throw new IllegalArgumentException(
                    "Expected logits [sequence length, vocabulary size] and at least two token IDs");
        }
        int vocabularySize = Math.toIntExact(logits.size(1));
        double total = 0.0;
        for (int position = 0; position < tokenIds.length - 1; position++) {
            int target = tokenIds[position + 1];
            if (target < 0 || target >= vocabularySize) {
                throw new IllegalArgumentException("Invalid target token ID: " + target);
            }
            double max = Double.NEGATIVE_INFINITY;
            for (int token = 0; token < vocabularySize; token++) {
                double score = logits.getDouble(position, token);
                if (!Double.isFinite(score)) {
                    throw new IllegalArgumentException("Logits must be finite");
                }
                max = Math.max(max, score);
            }
            double expSum = 0.0;
            for (int token = 0; token < vocabularySize; token++) {
                expSum += Math.exp(logits.getDouble(position, token) - max);
            }
            total += max + Math.log(expSum) - logits.getDouble(position, target);
        }
        return total / (tokenIds.length - 1);
    }
}
