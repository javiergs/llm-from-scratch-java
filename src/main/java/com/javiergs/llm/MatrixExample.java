package com.javiergs.llm;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

/**
 * Demonstrates a basic matrix multiplication using ND4J.
 *
 * @author Javier Gonzalez-Sanchez
 * @version 0.1.0 (2026-10-31)
 */
public final class MatrixExample {

    private MatrixExample() {
    }

    /**
     * Multiplies two small matrices and prints the result.
     *
     * @param args unused command-line arguments
     */
    public static void main(String[] args) {
        INDArray left = Nd4j.create(new float[][]{{1, 2}, {3, 4}});
        INDArray right = Nd4j.create(new float[][]{{5, 6}, {7, 8}});
        System.out.println(left.mmul(right));
    }
}
