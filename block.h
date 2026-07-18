// block.h
// El bloque transformer completo de GPT-2.
//   - gelu:        activacion no lineal
//   - feedforward: las dos capas lineales con GELU en medio
//   - bloque_transformer: atencion + feedforward con residuales y layernorm
//
// Con esto tienes UN bloque. GPT-2 apila 12 identicos.

#ifndef BLOCK_H
#define BLOCK_H

#include "matrix.h"
#include "layers.h"
#include "attention.h"
#include <cmath>

// ----------------------------------------------------------------------
// GELU (aproximacion con tanh, la que usa GPT-2)
// ----------------------------------------------------------------------
// GELU(x) = 0.5 * x * (1 + tanh( sqrt(2/pi) * (x + 0.044715 * x^3) ))
// Se aplica a cada elemento por separado.
Matrix gelu(const Matrix& x) {
    const float c = std::sqrt(2.0f / 3.14159265358979323846f); // sqrt(2/pi)
    Matrix salida(x.rows, x.cols);
    for (int i = 0; i < x.rows * x.cols; i++) {
        float v = x.data[i];
        float interno = c * (v + 0.044715f * v * v * v);
        salida.data[i] = 0.5f * v * (1.0f + std::tanh(interno));
    }
    return salida;
}

// ----------------------------------------------------------------------
// FEED-FORWARD (MLP del bloque)
// ----------------------------------------------------------------------
//   entrada:  [n x 768]  (ya pasada por layernorm ln_2)
//   c_fc_w:   [768 x 3072]   expande
//   c_fc_b:   [1 x 3072]
//   c_proj_w: [3072 x 768]   contrae
//   c_proj_b: [1 x 768]
//   salida:   [n x 768]
Matrix feedforward(const Matrix& entrada,
                   const Matrix& c_fc_w, const Matrix& c_fc_b,
                   const Matrix& c_proj_w, const Matrix& c_proj_b) {
    // 768 -> 3072
    Matrix h = lineal_conv1d(entrada, c_fc_w, c_fc_b);
    // activacion no lineal
    h = gelu(h);
    // 3072 -> 768
    return lineal_conv1d(h, c_proj_w, c_proj_b);
}

// ----------------------------------------------------------------------
// Pesos de un bloque transformer (los 12 tensores de una capa "h.N")
// ----------------------------------------------------------------------
struct PesosBloque {
    Matrix ln_1_w, ln_1_b;       // layernorm antes de atencion
    Matrix c_attn_w, c_attn_b;   // proyeccion QKV
    Matrix c_proj_w, c_proj_b;   // proyeccion de salida de atencion
    Matrix ln_2_w, ln_2_b;       // layernorm antes del feedforward
    Matrix c_fc_w, c_fc_b;       // feedforward expande
    Matrix mlp_proj_w, mlp_proj_b; // feedforward contrae
};

// ----------------------------------------------------------------------
// BLOQUE TRANSFORMER COMPLETO
// ----------------------------------------------------------------------
// Orden de GPT-2 (pre-norm), con conexiones residuales:
//   x = x + atencion(    layernorm_1(x) )
//   x = x + feedforward( layernorm_2(x) )
//
//   entrada: [n x 768]
//   salida:  [n x 768]
Matrix bloque_transformer(const Matrix& x, const PesosBloque& p, int num_cabezas) {
    // --- sub-bloque de atencion ---
    Matrix norm1 = layernorm(x, p.ln_1_w, p.ln_1_b);
    Matrix attn = multi_head_attention(
        norm1, p.c_attn_w, p.c_attn_b, p.c_proj_w, p.c_proj_b, num_cabezas);
    Matrix x1 = x.add(attn);   // residual: x + atencion(...)

    // --- sub-bloque de feedforward ---
    Matrix norm2 = layernorm(x1, p.ln_2_w, p.ln_2_b);
    Matrix ff = feedforward(
        norm2, p.c_fc_w, p.c_fc_b, p.mlp_proj_w, p.mlp_proj_b);
    Matrix x2 = x1.add(ff);    // residual: x1 + feedforward(...)

    return x2;
}

#endif // BLOCK_H
