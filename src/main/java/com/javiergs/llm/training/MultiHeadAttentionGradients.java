package com.javiergs.llm.training;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.NDArrayIndex;

/**
 * Differentiates multi-head causal attention with concatenation and output projection.
 * Mirrors the head slicing and scaled softmax in MultiHeadAttention.
 *
 * @author Javier Gonzalez-Sanchez
 * @version 0.1.0 (2026-10-31)
 */
public final class MultiHeadAttentionGradients {
    private MultiHeadAttentionGradients() {
    }

    /**
     * Gradients for input and the four projection matrices.
     *
     * @param inputGradient input derivative
     * @param queryWeightsGradient query weight derivative
     * @param keyWeightsGradient key weight derivative
     * @param valueWeightsGradient value weight derivative
     * @param outputWeightsGradient output projection derivative
     */
    public record Gradients(INDArray inputGradient, INDArray queryWeightsGradient,
                            INDArray keyWeightsGradient, INDArray valueWeightsGradient,
                            INDArray outputWeightsGradient) {
    }

    /**
     * Backpropagates through independent causal heads and their output projection.
     *
     * @param input token vectors [sequence, dimension]
     * @param queryWeights query projection [dimension, dimension]
     * @param keyWeights key projection [dimension, dimension]
     * @param valueWeights value projection [dimension, dimension]
     * @param outputWeights output projection [dimension, dimension]
     * @param heads positive head count dividing dimension
     * @param upstream output gradient [sequence, dimension]
     * @return all five gradients
     */
    public static Gradients backward(INDArray input, INDArray queryWeights, INDArray keyWeights,
                                     INDArray valueWeights, INDArray outputWeights,
                                     int heads, INDArray upstream) {
        if (input == null || queryWeights == null || keyWeights == null
                || valueWeights == null || outputWeights == null || upstream == null) {
            throw new NullPointerException("Matrices must not be null");
        }
        if (input.rank() != 2 || input.size(0) == 0 || input.size(1) == 0
                || upstream.rank() != 2 || upstream.size(0) != input.size(0)
                || upstream.size(1) != input.size(1)) {
            throw new IllegalArgumentException("Expected matching nonempty input and upstream matrices");
        }
        int dimension = Math.toIntExact(input.size(1));
        if (heads <= 0 || dimension % heads != 0) {
            throw new IllegalArgumentException("Head count must divide dimension");
        }
        for (INDArray weights : new INDArray[]{queryWeights, keyWeights, valueWeights, outputWeights}) {
            if (weights.rank() != 2 || weights.size(0) != dimension || weights.size(1) != dimension) {
                throw new IllegalArgumentException("Projection matrices must match model dimension");
            }
        }
        int length = Math.toIntExact(input.size(0));
        int headDimension = dimension / heads;
        INDArray queries = input.mmul(queryWeights);
        INDArray keys = input.mmul(keyWeights);
        INDArray values = input.mmul(valueWeights);
        INDArray concatenated = Nd4j.zeros(length, dimension);
        INDArray[] probabilities = new INDArray[heads];
        for (int head = 0; head < heads; head++) {
            int start = head * headDimension;
            INDArray q = slice(queries, start, headDimension);
            INDArray k = slice(keys, start, headDimension);
            INDArray v = slice(values, start, headDimension);
            INDArray scores = q.mmul(k.transpose()).div(Math.sqrt(headDimension));
            INDArray p = Nd4j.zeros(length, length);
            for (int row = 0; row < length; row++) {
                double max = Double.NEGATIVE_INFINITY;
                for (int col = 0; col <= row; col++) {
                    max = Math.max(max, scores.getDouble(row, col));
                }
                double sum = 0;
                for (int col = 0; col <= row; col++) {
                    sum += Math.exp(scores.getDouble(row, col) - max);
                }
                for (int col = 0; col <= row; col++) {
                    p.putScalar(row, col, Math.exp(scores.getDouble(row, col) - max) / sum);
                }
            }
            probabilities[head] = p;
            concatenated.get(NDArrayIndex.all(), NDArrayIndex.interval(start, start + headDimension))
                    .assign(p.mmul(v));
        }
        INDArray dOutputWeights = concatenated.transpose().mmul(upstream);
        INDArray dConcatenated = upstream.mmul(outputWeights.transpose());
        INDArray dQueries = Nd4j.zeros(length, dimension);
        INDArray dKeys = Nd4j.zeros(length, dimension);
        INDArray dValues = Nd4j.zeros(length, dimension);
        for (int head = 0; head < heads; head++) {
            int start = head * headDimension;
            INDArray q = slice(queries, start, headDimension);
            INDArray k = slice(keys, start, headDimension);
            INDArray v = slice(values, start, headDimension);
            INDArray p = probabilities[head];
            INDArray dHead = slice(dConcatenated, start, headDimension);
            INDArray dV = p.transpose().mmul(dHead);
            INDArray dP = dHead.mmul(v.transpose());
            INDArray dScores = Nd4j.zeros(length, length);
            for (int row = 0; row < length; row++) {
                double weighted = 0;
                for (int col = 0; col <= row; col++) {
                    weighted += p.getDouble(row, col) * dP.getDouble(row, col);
                }
                for (int col = 0; col <= row; col++) {
                    dScores.putScalar(row, col, p.getDouble(row, col)
                            * (dP.getDouble(row, col) - weighted));
                }
            }
            dQueries.get(NDArrayIndex.all(), NDArrayIndex.interval(start, start + headDimension))
                    .assign(dScores.mmul(k).div(Math.sqrt(headDimension)));
            dKeys.get(NDArrayIndex.all(), NDArrayIndex.interval(start, start + headDimension))
                    .assign(dScores.transpose().mmul(q).div(Math.sqrt(headDimension)));
            dValues.get(NDArrayIndex.all(), NDArrayIndex.interval(start, start + headDimension))
                    .assign(dV);
        }
        INDArray dInput = dQueries.mmul(queryWeights.transpose())
                .add(dKeys.mmul(keyWeights.transpose()))
                .add(dValues.mmul(valueWeights.transpose()));
        return new Gradients(dInput, input.transpose().mmul(dQueries),
                input.transpose().mmul(dKeys), input.transpose().mmul(dValues), dOutputWeights);
    }

    private static INDArray slice(INDArray matrix, int start, int width) {
        return matrix.get(NDArrayIndex.all(), NDArrayIndex.interval(start, start + width));
    }
}
