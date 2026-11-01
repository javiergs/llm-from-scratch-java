package com.javiergs.llm;

import org.junit.jupiter.api.Test;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import static org.junit.jupiter.api.Assertions.assertEquals;

/**
 * Verifies the ND4J numerical backend with a known matrix product.
 *
 * @author Javier Gonzalez-Sanchez
 * @version 0.1.0 (2026-10-31)
 */
class MatrixExampleTest {

    @Test
    void multipliesMatrices() {
        INDArray a = Nd4j.create(new float[][]{{1, 2}, {3, 4}});
        INDArray b = Nd4j.create(new float[][]{{5, 6}, {7, 8}});
        INDArray product = a.mmul(b);

        assertEquals(19.0, product.getDouble(0, 0), 1e-5);
        assertEquals(22.0, product.getDouble(0, 1), 1e-5);
        assertEquals(43.0, product.getDouble(1, 0), 1e-5);
        assertEquals(50.0, product.getDouble(1, 1), 1e-5);
    }
}
