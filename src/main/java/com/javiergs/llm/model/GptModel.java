package com.javiergs.llm.model;

import com.javiergs.llm.embedding.PositionalEmbedding;
import com.javiergs.llm.embedding.TokenEmbedding;
import com.javiergs.llm.transformer.LayerNormalization;
import com.javiergs.llm.transformer.TransformerBlock;
import java.util.Random;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

/**
 * A small decoder-only GPT-style model with a forward pass that produces
 * unnormalized next-token scores (logits) at every input position.
 * Parameters are initialized randomly and are not trained.
 *
 * @author Javier Gonzalez-Sanchez
 * @version 0.1.0 (2026-10-31)
 */
public final class GptModel {
    private final int vocabularySize;
    private final int contextLength;
    private final int dimension;
    private final TokenEmbedding tokens;
    private final PositionalEmbedding positions;
    private final TransformerBlock[] blocks;
    private final LayerNormalization finalNormalization;
    private final INDArray outputWeights;

    /**
     * Creates an untrained decoder-only transformer.
     *
     * @param vocabularySize number of distinct token IDs
     * @param contextLength maximum sequence length
     * @param dimension model embedding width
     * @param heads attention head count
     * @param layers number of transformer blocks
     * @param hiddenDimension feed-forward intermediate width
     * @param seed reproducible initialization seed
     */
    public GptModel(int vocabularySize, int contextLength, int dimension,
                    int heads, int layers, int hiddenDimension, long seed) {
        if (vocabularySize <= 0 || contextLength <= 0 || dimension <= 0
                || heads <= 0 || dimension % heads != 0
                || layers <= 0 || hiddenDimension <= 0) {
            throw new IllegalArgumentException("All sizes must be positive; dimension must be divisible by heads");
        }
        this.vocabularySize = vocabularySize;
        this.contextLength = contextLength;
        this.dimension = dimension;
        tokens = new TokenEmbedding(vocabularySize, dimension, seed);
        positions = new PositionalEmbedding(contextLength, dimension, seed + 1);
        blocks = new TransformerBlock[layers];
        for (int i = 0; i < layers; i++) {
            blocks[i] = new TransformerBlock(dimension, heads, hiddenDimension, seed + 2 + i);
        }
        finalNormalization = new LayerNormalization(dimension, 1e-5);
        Random random = new Random(seed + 2 + layers);
        float[][] values = new float[dimension][vocabularySize];
        double scale = 1.0 / Math.sqrt(dimension);
        for (int i = 0; i < dimension; i++) {
            for (int j = 0; j < vocabularySize; j++) {
                values[i][j] = (float) (random.nextGaussian() * scale);
            }
        }
        outputWeights = Nd4j.create(values);
    }

    /**
     * Computes vocabulary logits for every position in a nonempty token sequence.
     *
     * @param tokenIds input token IDs
     * @return logits of shape [sequence length, vocabulary size]
     */
    public INDArray forward(int[] tokenIds) {
        if (tokenIds == null) {
            throw new NullPointerException("tokenIds");
        }
        if (tokenIds.length == 0 || tokenIds.length > contextLength) {
            throw new IllegalArgumentException("Input length must be between 1 and " + contextLength);
        }
        INDArray state = positions.addTo(tokens.forward(tokenIds));
        for (TransformerBlock block : blocks) {
            state = block.forward(state);
        }
        return finalNormalization.forward(state).mmul(outputWeights);
    }

    /**
     * Returns the number of vocabulary scores per position.
     *
     * @return vocabulary size
     */
    public int vocabularySize() {
        return vocabularySize;
    }

    /**
     * Returns the model embedding width.
     *
     * @return embedding dimension
     */
    public int dimension() {
        return dimension;
    }
    /**
     * Returns independent copies of all trainable matrices. The order is
     * token embeddings, positional embeddings, six projections per block,
     * and finally the vocabulary output projection.
     *
     * @return model parameter copies
     */
    public INDArray[] parameters() {
        java.util.List<INDArray> result = new java.util.ArrayList<>();
        result.add(tokens.weights());
        result.add(positions.weights());
        for (TransformerBlock block : blocks) {
            java.util.Collections.addAll(result, block.parameters());
        }
        result.add(outputWeights.dup());
        return result.toArray(new INDArray[0]);
    }

    /**
     * Replaces model weights using shape-validated defensive copies.
     * Intended for educational optimizers in later lessons.
     *
     * @param parameters weights in the order returned by parameters()
     */
    public void setParameters(INDArray[] parameters) {
        INDArray[] current = parameters();
        if (parameters == null || parameters.length != current.length) {
            throw new IllegalArgumentException("Incorrect model parameter count");
        }
        for (int i = 0; i < current.length; i++) {
            if (parameters[i] == null || !java.util.Arrays.equals(
                    parameters[i].shape(), current[i].shape())) {
                throw new IllegalArgumentException("Incorrect parameter shape at index " + i);
            }
        }
        tokens.setWeights(parameters[0]);
        positions.setWeights(parameters[1]);
        for (int i = 0; i < blocks.length; i++) {
            blocks[i].setParameters(java.util.Arrays.copyOfRange(parameters,
                    2 + 6 * i, 2 + 6 * (i + 1)));
        }
        outputWeights.assign(parameters[parameters.length - 1]);
    }

}
