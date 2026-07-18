// attention.h
// Self-attention causal (una sola cabeza por ahora).
// Usa piezas que ya construimos y probamos: matmul, transpose, softmax.
// Lo unico nuevo aqui es la mascara causal.
//
// Flujo de una cabeza:
//   1. scores = Q * K^T
//   2. escalar: scores / sqrt(dim_cabeza)
//   3. mascara causal: poner -infinito donde j > i (el futuro)
//   4. softmax por filas
//   5. salida = pesos * V

#ifndef ATTENTION_H
#define ATTENTION_H

#include "matrix.h"
#include "softmax.h"
#include <cmath>
#include <limits>

// Aplica la mascara causal EN SITIO sobre la matriz de scores [n x n].
// Regla: la posicion [i][j] se enmascara si j > i (columna en el futuro).
// Usamos un numero muy negativo en vez de -infinito real para evitar
// problemas de aritmetica; tras el softmax queda en ~0 igual.
void aplicar_mascara_causal(Matrix& scores) {
    const float NEG = -1e10f;
    for (int i = 0; i < scores.rows; i++)
        for (int j = 0; j < scores.cols; j++)
            if (j > i)
                scores.at(i, j) = NEG;
}

// Una cabeza de atencion.
//   Q, K, V: cada uno [n x dim_cabeza]  (ya proyectados)
//   devuelve: [n x dim_cabeza]
//
// n = numero de tokens, dim_cabeza = 64 en GPT-2 (768 / 12 cabezas)
Matrix atencion_una_cabeza(const Matrix& Q, const Matrix& K, const Matrix& V) {
    int dim_cabeza = Q.cols;

    // 1. scores = Q * K^T  -> [n x n]
    //    cada entrada (i,j) = producto punto de la query i con la key j
    Matrix scores = Q.matmul(K.transpose());

    // 2. escalar por 1/sqrt(dim_cabeza)
    float escala = 1.0f / std::sqrt((float)dim_cabeza);
    for (int i = 0; i < scores.rows; i++)
        for (int j = 0; j < scores.cols; j++)
            scores.at(i, j) *= escala;

    // 3. mascara causal (no ver el futuro)
    aplicar_mascara_causal(scores);

    // 4. softmax por filas -> pesos de atencion [n x n], cada fila suma 1
    Matrix pesos = softmax(scores);

    // 5. salida = pesos * V  -> [n x dim_cabeza]
    //    cada token se lleva una mezcla ponderada de los values
    return pesos.matmul(V);
}

// ----------------------------------------------------------------------
// CAPA LINEAL estilo Conv1D de GPT-2
// ----------------------------------------------------------------------
// OJO: este es el detalle del layout que veniamos avisando.
// GPT-2 guarda los pesos (c_attn, c_proj, c_fc) ya traspuestos (Conv1D),
// asi que la operacion es  salida = entrada * W + bias  (SIN transponer W).
//
//   entrada: [n x d_in]
//   W:       [d_in x d_out]   (tal cual viene del safetensors)
//   bias:    [1 x d_out]
//   salida:  [n x d_out]
Matrix lineal_conv1d(const Matrix& entrada, const Matrix& W, const Matrix& bias) {
    Matrix salida = entrada.matmul(W);      // [n x d_out]
    // sumar el bias a cada fila
    for (int i = 0; i < salida.rows; i++)
        for (int j = 0; j < salida.cols; j++)
            salida.at(i, j) += bias.at(0, j);
    return salida;
}

// Extrae un bloque de columnas [col_ini, col_ini + ancho) de una matriz.
// Sirve para partir Q/K/V en cabezas, y para separar el QKV combinado.
Matrix sub_columnas(const Matrix& m, int col_ini, int ancho) {
    Matrix res(m.rows, ancho);
    for (int i = 0; i < m.rows; i++)
        for (int j = 0; j < ancho; j++)
            res.at(i, j) = m.at(i, col_ini + j);
    return res;
}

// ----------------------------------------------------------------------
// MULTI-HEAD ATTENTION completa (el bloque de atencion de GPT-2)
// ----------------------------------------------------------------------
//   entrada:      [n x 768]  (ya pasada por layernorm ln_1)
//   c_attn_w:     [768 x 2304]  proyeccion combinada Q,K,V
//   c_attn_b:     [1 x 2304]
//   c_proj_w:     [768 x 768]   proyeccion de salida
//   c_proj_b:     [1 x 768]
//   num_cabezas:  12 en GPT-2
//   devuelve:     [n x 768]
Matrix multi_head_attention(const Matrix& entrada,
                            const Matrix& c_attn_w, const Matrix& c_attn_b,
                            const Matrix& c_proj_w, const Matrix& c_proj_b,
                            int num_cabezas) {
    int n = entrada.rows;
    int d = entrada.cols;               // 768
    int dim_cabeza = d / num_cabezas;   // 64

    // 1. Proyeccion QKV combinada -> [n x 2304]
    Matrix qkv = lineal_conv1d(entrada, c_attn_w, c_attn_b);

    // 2. Separar en Q, K, V -> cada uno [n x 768]
    Matrix Q = sub_columnas(qkv, 0,     d);
    Matrix K = sub_columnas(qkv, d,     d);
    Matrix V = sub_columnas(qkv, 2 * d, d);

    // 3. Para cada cabeza: tomar su trozo de 64 columnas y correr atencion.
    //    Ir escribiendo el resultado en la matriz concatenada.
    Matrix concat(n, d);
    for (int h = 0; h < num_cabezas; h++) {
        int col = h * dim_cabeza;
        Matrix Qh = sub_columnas(Q, col, dim_cabeza);
        Matrix Kh = sub_columnas(K, col, dim_cabeza);
        Matrix Vh = sub_columnas(V, col, dim_cabeza);

        Matrix salida_h = atencion_una_cabeza(Qh, Kh, Vh);  // [n x 64]

        // pegar en las columnas correspondientes de concat
        for (int i = 0; i < n; i++)
            for (int j = 0; j < dim_cabeza; j++)
                concat.at(i, col + j) = salida_h.at(i, j);
    }

    // 4. Proyeccion de salida -> [n x 768]
    return lineal_conv1d(concat, c_proj_w, c_proj_b);
}

#endif // ATTENTION_H
