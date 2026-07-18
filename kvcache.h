// kvcache.h
// FASE 1 - Optimizacion: KV cache.
//
// EL PROBLEMA
// Sin cache, para generar el token N el programa recalcula las keys y values
// de los N-1 tokens anteriores. Pero esos tokens no cambiaron: por la mascara
// causal, un token solo ve al pasado, asi que su representacion es la misma
// en el paso 1 que en el paso 30.
//
// LA IDEA
// Guardamos las K y V de cada token ya procesado. Cuando llega uno nuevo:
//   1. calculamos Q, K, V solo del token nuevo
//   2. agregamos su K y V al cache
//   3. la atencion usa la Q nueva contra TODAS las K del cache
//
// Se cachean K y V (no Q) porque necesitamos la query solo del ultimo token
// -es el que predice-, pero necesitamos las keys/values de todos, porque el
// token nuevo debe poder mirar todo el pasado.
//
// COSTO
//   sin cache: cada paso procesa n tokens  -> O(n^2) en total
//   con cache: cada paso procesa 1 token   -> O(n)   en total

#ifndef KVCACHE_H
#define KVCACHE_H

#include "matrix.h"
#include "layers.h"
#include "softmax.h"
#include "attention.h"
#include "block.h"
#include <vector>
#include <cmath>

// ----------------------------------------------------------------------
// CACHE DE UNA CAPA
// ----------------------------------------------------------------------
// Guarda las K y V acumuladas de todos los tokens procesados hasta ahora.
// Cada matriz crece una fila por cada token nuevo:
//   K: [tokens_hasta_ahora x 768]
//   V: [tokens_hasta_ahora x 768]
struct CacheCapa {
    Matrix K;
    Matrix V;
    int num_tokens = 0;   // cuantas filas validas hay

    // Reserva espacio de golpe para evitar realocaciones constantes.
    // max_tokens es el limite de contexto (1024 en GPT-2).
    void reservar(int max_tokens, int d_model) {
        K = Matrix(max_tokens, d_model);
        V = Matrix(max_tokens, d_model);
        num_tokens = 0;
    }

    // Agrega las K y V de un token nuevo (cada una es un vector [1 x 768])
    void agregar(const Matrix& k_nueva, const Matrix& v_nueva) {
        int d = k_nueva.cols;
        for (int j = 0; j < d; j++) {
            K.at(num_tokens, j) = k_nueva.at(0, j);
            V.at(num_tokens, j) = v_nueva.at(0, j);
        }
        num_tokens++;
    }

    void limpiar() { num_tokens = 0; }
};

// ----------------------------------------------------------------------
// CACHE COMPLETO (una entrada por cada capa del modelo)
// ----------------------------------------------------------------------
struct KVCache {
    std::vector<CacheCapa> capas;

    void inicializar(int num_capas, int max_tokens, int d_model) {
        capas.resize(num_capas);
        for (auto& c : capas) c.reservar(max_tokens, d_model);
    }

    void limpiar() {
        for (auto& c : capas) c.limpiar();
    }

    int tokens_procesados() const {
        return capas.empty() ? 0 : capas[0].num_tokens;
    }
};

// ----------------------------------------------------------------------
// ATENCION DE UNA CABEZA CON CACHE
// ----------------------------------------------------------------------
//   Q_nueva: [1 x dim_cabeza]        query del token nuevo, solo una fila
//   K_todas: [n x dim_cabeza]        keys de TODOS los tokens (del cache)
//   V_todas: [n x dim_cabeza]        values de TODOS los tokens (del cache)
//   devuelve: [1 x dim_cabeza]
//
// OJO: aqui NO hay mascara causal. Como el cache solo contiene tokens
// anteriores (mas el actual), no hay futuro que enmascarar. El token nuevo
// puede mirar legitimamente todo lo que hay en el cache.
Matrix atencion_cabeza_cache(const Matrix& Q_nueva,
                             const Matrix& K_todas,
                             const Matrix& V_todas,
                             int n_validos) {
    int dim_cabeza = Q_nueva.cols;
    float escala = 1.0f / std::sqrt((float)dim_cabeza);

    // 1. scores = Q_nueva * K^T, pero solo contra las n_validos filas reales.
    //    Resultado: [1 x n_validos]
    Matrix scores(1, n_validos);
    for (int j = 0; j < n_validos; j++) {
        float suma = 0.0f;
        for (int k = 0; k < dim_cabeza; k++)
            suma += Q_nueva.at(0, k) * K_todas.at(j, k);
        scores.at(0, j) = suma * escala;
    }

    // 2. softmax sobre la unica fila
    Matrix pesos = softmax(scores);

    // 3. salida = pesos * V  -> [1 x dim_cabeza]
    Matrix salida(1, dim_cabeza);
    for (int k = 0; k < dim_cabeza; k++) {
        float suma = 0.0f;
        for (int j = 0; j < n_validos; j++)
            suma += pesos.at(0, j) * V_todas.at(j, k);
        salida.at(0, k) = suma;
    }
    return salida;
}

// ----------------------------------------------------------------------
// MULTI-HEAD ATTENTION CON CACHE
// ----------------------------------------------------------------------
//   entrada: [1 x 768]  el token nuevo, ya pasado por layernorm
//   cache:   el cache de ESTA capa (se actualiza dentro)
//   devuelve: [1 x 768]
Matrix multi_head_cache(const Matrix& entrada,
                        const Matrix& c_attn_w, const Matrix& c_attn_b,
                        const Matrix& c_proj_w, const Matrix& c_proj_b,
                        int num_cabezas,
                        CacheCapa& cache) {
    int d = entrada.cols;               // 768
    int dim_cabeza = d / num_cabezas;   // 64

    // 1. Proyeccion QKV del token nuevo -> [1 x 2304]
    Matrix qkv = lineal_conv1d(entrada, c_attn_w, c_attn_b);

    // 2. Separar Q, K, V -> cada uno [1 x 768]
    Matrix Q = sub_columnas(qkv, 0,     d);
    Matrix K = sub_columnas(qkv, d,     d);
    Matrix V = sub_columnas(qkv, 2 * d, d);

    // 3. Guardar la K y V nuevas en el cache
    cache.agregar(K, V);
    int n = cache.num_tokens;   // cuantos tokens hay en total ahora

    // 4. Para cada cabeza: su trozo de Q contra su trozo de K/V del cache
    Matrix concat(1, d);
    for (int h = 0; h < num_cabezas; h++) {
        int col = h * dim_cabeza;

        // Q de esta cabeza (solo el token nuevo)
        Matrix Qh = sub_columnas(Q, col, dim_cabeza);

        // K y V de esta cabeza, de TODOS los tokens del cache.
        // Extraemos las columnas [col, col+64) de las n filas validas.
        Matrix Kh(n, dim_cabeza);
        Matrix Vh(n, dim_cabeza);
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < dim_cabeza; j++) {
                Kh.at(i, j) = cache.K.at(i, col + j);
                Vh.at(i, j) = cache.V.at(i, col + j);
            }
        }

        Matrix salida_h = atencion_cabeza_cache(Qh, Kh, Vh, n);

        for (int j = 0; j < dim_cabeza; j++)
            concat.at(0, col + j) = salida_h.at(0, j);
    }

    // 5. Proyeccion de salida
    return lineal_conv1d(concat, c_proj_w, c_proj_b);
}

// ----------------------------------------------------------------------
// BLOQUE TRANSFORMER CON CACHE
// ----------------------------------------------------------------------
// Identico al bloque normal, solo cambia la atencion.
//   x: [1 x 768]  (un solo token)
Matrix bloque_cache(const Matrix& x, const PesosBloque& p,
                    int num_cabezas, CacheCapa& cache) {
    Matrix norm1 = layernorm(x, p.ln_1_w, p.ln_1_b);
    Matrix attn = multi_head_cache(
        norm1, p.c_attn_w, p.c_attn_b, p.c_proj_w, p.c_proj_b,
        num_cabezas, cache);
    Matrix x1 = x.add(attn);

    Matrix norm2 = layernorm(x1, p.ln_2_w, p.ln_2_b);
    Matrix ff = feedforward(
        norm2, p.c_fc_w, p.c_fc_b, p.mlp_proj_w, p.mlp_proj_b);
    return x1.add(ff);
}

#endif // KVCACHE_H
