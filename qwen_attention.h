// qwen_attention.h
// FASE 3 - Atencion con GQA (Grouped Query Attention) y RoPE.
//
// LA DIFERENCIA CON GPT-2
// En GPT-2, 12 cabezas de query tenian 12 cabezas de key/value: una
// cada una. En Qwen2.5-0.5B hay 14 cabezas de query pero solo 2 de
// key/value. Cada grupo de 7 queries COMPARTE las mismas K y V.
//
//   GPT-2:  Q[12 cabezas]  K[12 cabezas]  V[12 cabezas]
//   Qwen:   Q[14 cabezas]  K[2 cabezas]   V[2 cabezas]
//                          \___ compartidas por 7 queries cada una
//
// POR QUE
// El KV cache se vuelve 7 veces mas chico. Con 32768 de contexto, esa
// diferencia es la que hace viable el modelo: cachear 14 cabezas de
// K y V por cada uno de 32k tokens seria prohibitivo en memoria.
//
// La calidad casi no sufre: resulta que las queries necesitan
// diversidad (por eso 14 distintas), pero las keys y values toleran
// ser compartidas.

#ifndef QWEN_ATTENTION_H
#define QWEN_ATTENTION_H

#include "matrix.h"
#include "softmax.h"
#include "paralelo.h"
#include "qwen_ops.h"
#include <cmath>
#include <vector>

// ----------------------------------------------------------------------
// CACHE PARA GQA
// ----------------------------------------------------------------------
// Guarda K y V de las cabezas KV (pocas), no de las de query.
// Dimension por token: num_cabezas_kv * dim_cabeza
//   Qwen2.5-0.5B: 2 * 64 = 128  (contra 896 si no hubiera GQA)
struct CacheQwen {
    Matrix K;   // [max_tokens x (n_kv * dim_cabeza)]
    Matrix V;
    int num_tokens = 0;

    void reservar(int max_tokens, int dim_kv) {
        K = Matrix(max_tokens, dim_kv);
        V = Matrix(max_tokens, dim_kv);
        num_tokens = 0;
    }

    void agregar(const float* k, const float* v, int dim_kv) {
        for (int j = 0; j < dim_kv; j++) {
            K.at(num_tokens, j) = k[j];
            V.at(num_tokens, j) = v[j];
        }
        num_tokens++;
    }

    void limpiar() { num_tokens = 0; }
};

struct KVCacheQwen {
    std::vector<CacheQwen> capas;

    void inicializar(int num_capas, int max_tokens, int dim_kv) {
        capas.resize(num_capas);
        for (auto& c : capas) c.reservar(max_tokens, dim_kv);
    }
    void limpiar() { for (auto& c : capas) c.limpiar(); }
    int tokens_procesados() const {
        return capas.empty() ? 0 : capas[0].num_tokens;
    }
};

// ----------------------------------------------------------------------
// Capa lineal con bias opcional
// ----------------------------------------------------------------------
// En Qwen, los pesos vienen del GGUF con forma [salida, entrada] en
// nuestra Matrix (filas=dims[1], cols=dims[0]). Para calcular
// salida = entrada * W^T usamos matmul_transpuesta_paralelo, que evita
// construir la transpuesta.
//
//   entrada: [n x d_in]
//   W:       [d_out x d_in]   (tal cual sale del GGUF)
//   bias:    [1 x d_out] o vacia
//   salida:  [n x d_out]
inline Matrix lineal_qwen(const Matrix& entrada, const Matrix& W,
                          const Matrix* bias = nullptr) {
    Matrix salida = matmul_transpuesta_paralelo(entrada, W);
    if (bias && bias->cols == salida.cols) {
        for (int i = 0; i < salida.rows; i++)
            for (int j = 0; j < salida.cols; j++)
                salida.at(i, j) += bias->at(0, j);
    }
    return salida;
}

// ----------------------------------------------------------------------
// ATENCION GQA CON CACHE (procesa un token)
// ----------------------------------------------------------------------
//   x:          [1 x d_model]  el token ya normalizado
//   Wq, Wk, Wv: pesos de proyeccion (con sus bias, que Qwen si tiene)
//   Wo:         proyeccion de salida (sin bias en Qwen)
//   n_heads:    cabezas de query (14)
//   n_kv:       cabezas de key/value (2)
//   pos:        posicion del token en la secuencia (para RoPE)
//   cache:      se actualiza dentro
//   devuelve:   [1 x d_model]
Matrix atencion_qwen(const Matrix& x,
                     const Matrix& Wq, const Matrix& bq,
                     const Matrix& Wk, const Matrix& bk,
                     const Matrix& Wv, const Matrix& bv,
                     const Matrix& Wo,
                     int n_heads, int n_kv,
                     int pos, float rope_base,
                     CacheQwen& cache) {
    int d_model = x.cols;
    int dim_cabeza = d_model / n_heads;      // 896 / 14 = 64
    int dim_kv = n_kv * dim_cabeza;          // 2 * 64 = 128
    int grupo = n_heads / n_kv;              // 14 / 2 = 7 queries por cabeza kv

    // 1. Proyecciones. Q es grande, K y V son chicas (esa es la gracia de GQA)
    Matrix Q = lineal_qwen(x, Wq, &bq);   // [1 x 896]
    Matrix K = lineal_qwen(x, Wk, &bk);   // [1 x 128]
    Matrix V = lineal_qwen(x, Wv, &bv);   // [1 x 128]

    // 2. RoPE sobre Q y K (NO sobre V: los values no llevan posicion).
    //    Se aplica por cabeza, sobre cada trozo de 64 dimensiones.
    for (int h = 0; h < n_heads; h++)
        aplicar_rope_fila(&Q.data[h * dim_cabeza], dim_cabeza, pos, rope_base);
    for (int h = 0; h < n_kv; h++)
        aplicar_rope_fila(&K.data[h * dim_cabeza], dim_cabeza, pos, rope_base);

    // 3. Guardar K y V (ya rotadas) en el cache
    cache.agregar(K.data.data(), V.data.data(), dim_kv);
    int n = cache.num_tokens;

    // 4. Atencion por cada cabeza de query
    float escala = 1.0f / std::sqrt((float)dim_cabeza);
    Matrix concat(1, d_model);

    for (int h = 0; h < n_heads; h++) {
        // Que cabeza KV le toca a esta query
        int h_kv = h / grupo;
        int off_q  = h * dim_cabeza;
        int off_kv = h_kv * dim_cabeza;

        // scores contra todos los tokens del cache
        std::vector<float> scores(n);
        for (int t = 0; t < n; t++) {
            float s = 0.0f;
            for (int j = 0; j < dim_cabeza; j++)
                s += Q.data[off_q + j] * cache.K.at(t, off_kv + j);
            scores[t] = s * escala;
        }

        // softmax estable (restar el maximo)
        float mx = scores[0];
        for (int t = 1; t < n; t++) if (scores[t] > mx) mx = scores[t];
        float suma = 0.0f;
        for (int t = 0; t < n; t++) {
            scores[t] = std::exp(scores[t] - mx);
            suma += scores[t];
        }
        for (int t = 0; t < n; t++) scores[t] /= suma;

        // mezcla ponderada de los values
        for (int j = 0; j < dim_cabeza; j++) {
            float acc = 0.0f;
            for (int t = 0; t < n; t++)
                acc += scores[t] * cache.V.at(t, off_kv + j);
            concat.at(0, off_q + j) = acc;
        }
    }

    // 5. Proyeccion de salida (Qwen no usa bias aqui)
    return lineal_qwen(concat, Wo, nullptr);
}

#endif // QWEN_ATTENTION_H
