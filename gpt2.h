// gpt2.h
// El modelo GPT-2 completo. Carga todos los pesos del safetensors y
// ejecuta el forward pass entero: tokens -> logits.
//
// Flujo:
//   embeddings -> 12 bloques transformer -> layernorm final -> logits
//
// La proyeccion final usa "weight tying": reutiliza wte transpuesto
// para pasar de 768 dimensiones a los 50257 logits del vocabulario.

#ifndef GPT2_H
#define GPT2_H

#include "matrix.h"
#include "layers.h"
#include "attention.h"
#include "block.h"
#include "safetensors.h"
#include <vector>
#include <string>
#include <iostream>

class GPT2 {
public:
    // Hiperparametros de GPT-2 small (124M)
    int num_capas = 12;
    int num_cabezas = 12;
    int d_model = 768;
    int vocab = 50257;

    Matrix wte;   // token embeddings   [50257 x 768]
    Matrix wpe;   // positional embeddings [1024 x 768]
    Matrix ln_f_w, ln_f_b;  // layernorm final
    std::vector<PesosBloque> bloques;  // los 12 bloques

    // Carga todos los pesos desde el archivo safetensors
    void cargar(const std::string& ruta) {
        SafeTensors st(ruta);

        std::cout << "Cargando embeddings...\n";
        wte = st.getTensor("wte.weight");
        wpe = st.getTensor("wpe.weight");

        std::cout << "Cargando layernorm final...\n";
        ln_f_w = st.getTensor("ln_f.weight");
        ln_f_b = st.getTensor("ln_f.bias");

        std::cout << "Cargando " << num_capas << " bloques transformer...\n";
        bloques.resize(num_capas);
        for (int i = 0; i < num_capas; i++) {
            std::string pre = "h." + std::to_string(i) + ".";
            PesosBloque& p = bloques[i];

            p.ln_1_w = st.getTensor(pre + "ln_1.weight");
            p.ln_1_b = st.getTensor(pre + "ln_1.bias");

            p.c_attn_w = st.getTensor(pre + "attn.c_attn.weight");
            p.c_attn_b = st.getTensor(pre + "attn.c_attn.bias");
            p.c_proj_w = st.getTensor(pre + "attn.c_proj.weight");
            p.c_proj_b = st.getTensor(pre + "attn.c_proj.bias");

            p.ln_2_w = st.getTensor(pre + "ln_2.weight");
            p.ln_2_b = st.getTensor(pre + "ln_2.bias");

            // OJO: attn.c_proj va en c_proj_*, pero mlp.c_proj va en mlp_proj_*
            p.c_fc_w = st.getTensor(pre + "mlp.c_fc.weight");
            p.c_fc_b = st.getTensor(pre + "mlp.c_fc.bias");
            p.mlp_proj_w = st.getTensor(pre + "mlp.c_proj.weight");
            p.mlp_proj_b = st.getTensor(pre + "mlp.c_proj.bias");
        }
        std::cout << "Modelo cargado.\n";
    }

    // Forward pass completo.
    //   tokens: lista de token IDs
    //   devuelve: logits [n x 50257]  (una fila de puntajes por cada token)
    Matrix forward(const std::vector<int>& tokens) {
        // 1. embeddings: tokens -> [n x 768]
        Matrix x = embeddings(tokens, wte, wpe);

        // 2. pasar por los 12 bloques, cada salida alimenta al siguiente
        for (int i = 0; i < num_capas; i++)
            x = bloque_transformer(x, bloques[i], num_cabezas);

        // 3. layernorm final
        x = layernorm(x, ln_f_w, ln_f_b);

        // 4. proyeccion a logits usando weight tying (wte transpuesto)
        //    [n x 768] * [768 x 50257] = [n x 50257]
        Matrix logits = x.matmul(wte.transpose());

        return logits;
    }

    // Utilidad: dado los logits, devuelve el token de mayor puntaje
    // en la ULTIMA posicion (la prediccion del siguiente token).
    int argmax_ultima_fila(const Matrix& logits) {
        int fila = logits.rows - 1;
        int mejor = 0;
        float mejor_val = logits.at(fila, 0);
        for (int j = 1; j < logits.cols; j++) {
            if (logits.at(fila, j) > mejor_val) {
                mejor_val = logits.at(fila, j);
                mejor = j;
            }
        }
        return mejor;
    }
};

#endif // GPT2_H
