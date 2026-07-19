// qwen.h
// FASE 3 - El modelo Qwen2.5 completo, cargado desde GGUF.
//
// ARQUITECTURA (comparada con GPT-2)
//
//   GPT-2                          Qwen2.5
//   -----                          -------
//   wte + wpe (posicion sumada)    token_embd + RoPE (posicion rotada)
//   LayerNorm                      RMSNorm
//   12 cabezas Q / 12 KV           14 cabezas Q / 2 KV  (GQA)
//   c_fc -> GELU -> c_proj         ffn_gate/ffn_up -> SiLU -> ffn_down
//   weight tying (reusa wte)       output.weight separado
//
// FLUJO DE UN BLOQUE
//   x = x + atencion( rmsnorm(x, attn_norm) )
//   x = x + swiglu(   rmsnorm(x, ffn_norm)  )

#ifndef QWEN_H
#define QWEN_H

#include "matrix.h"
#include "gguf.h"
#include "qwen_ops.h"
#include "qwen_attention.h"
#include "paralelo.h"
#include "bpe.h"
#include <vector>
#include <string>
#include <iostream>

// Pesos de un bloque transformer de Qwen
struct BloqueQwen {
    Matrix attn_norm;        // RMSNorm antes de la atencion
    Matrix wq, bq;           // query  (Qwen SI tiene bias en q/k/v)
    Matrix wk, bk;           // key
    Matrix wv, bv;           // value
    Matrix wo;               // salida de atencion (sin bias)
    Matrix ffn_norm;         // RMSNorm antes del feed-forward
    Matrix ffn_gate;         // compuerta
    Matrix ffn_up;           // camino principal
    Matrix ffn_down;         // proyeccion de vuelta
};

class Qwen {
public:
    // Hiperparametros (se leen del GGUF)
    int num_capas = 0;
    int d_model = 0;
    int n_heads = 0;
    int n_kv = 0;
    int d_ff = 0;
    int vocab = 0;
    int contexto_max = 0;
    float rope_base = 1000000.0f;
    float eps = 1e-6f;

    Matrix token_embd;       // [vocab x d_model]
    Matrix output_norm;      // RMSNorm final
    Matrix output_w;         // proyeccion a logits [vocab x d_model]
    bool usa_weight_tying = false;

    std::vector<BloqueQwen> bloques;

    // Vocabulario y tokenizador
    std::vector<std::string> tokens_texto;
    BPE bpe;
    int bos = -1, eos = -1;

    int dim_cabeza() const { return d_model / n_heads; }
    int dim_kv() const { return n_kv * dim_cabeza(); }

    // ------------------------------------------------------------------
    void cargar(const std::string& ruta) {
        GGUF g(ruta);

        if (g.arquitectura() != "qwen2")
            std::cerr << "Aviso: arquitectura '" << g.arquitectura()
                      << "' (esperada 'qwen2'). Intentando de todos modos.\n";

        // --- Hiperparametros ---
        num_capas    = (int)g.hiper("block_count");
        d_model      = (int)g.hiper("embedding_length");
        n_heads      = (int)g.hiper("attention.head_count");
        n_kv         = (int)g.hiper("attention.head_count_kv");
        d_ff         = (int)g.hiper("feed_forward_length");
        contexto_max = (int)g.hiper("context_length");
        rope_base    = (float)g.real(g.arquitectura() + ".rope.freq_base",
                                     1000000.0);
        eps = (float)g.real(
            g.arquitectura() + ".attention.layer_norm_rms_epsilon", 1e-6);

        std::cout << "Arquitectura: " << g.arquitectura() << "\n";
        std::cout << "  capas=" << num_capas
                  << " d_model=" << d_model
                  << " heads=" << n_heads << "/" << n_kv << " (GQA)"
                  << " d_ff=" << d_ff << "\n";
        std::cout << "  contexto=" << contexto_max
                  << " rope_base=" << rope_base
                  << " eps=" << eps << "\n";

        // --- Embeddings ---
        std::cout << "Cargando embeddings...\n";
        token_embd = g.leer_tensor("token_embd.weight");
        vocab = token_embd.rows;

        // --- Normalizacion y salida ---
        output_norm = g.leer_tensor("output_norm.weight");

        // Algunos modelos comparten el embedding con la salida
        if (g.tiene_tensor("output.weight")) {
            std::cout << "Cargando proyeccion de salida...\n";
            output_w = g.leer_tensor("output.weight");
            usa_weight_tying = false;
        } else {
            std::cout << "Sin output.weight: usa weight tying\n";
            usa_weight_tying = true;
        }

        // --- Bloques ---
        std::cout << "Cargando " << num_capas << " bloques...\n";
        bloques.resize(num_capas);
        for (int i = 0; i < num_capas; i++) {
            std::string p = "blk." + std::to_string(i) + ".";
            BloqueQwen& b = bloques[i];

            b.attn_norm = g.leer_tensor(p + "attn_norm.weight");
            b.wq = g.leer_tensor(p + "attn_q.weight");
            b.bq = g.leer_tensor(p + "attn_q.bias");
            b.wk = g.leer_tensor(p + "attn_k.weight");
            b.bk = g.leer_tensor(p + "attn_k.bias");
            b.wv = g.leer_tensor(p + "attn_v.weight");
            b.bv = g.leer_tensor(p + "attn_v.bias");
            b.wo = g.leer_tensor(p + "attn_output.weight");

            b.ffn_norm = g.leer_tensor(p + "ffn_norm.weight");
            b.ffn_gate = g.leer_tensor(p + "ffn_gate.weight");
            b.ffn_up   = g.leer_tensor(p + "ffn_up.weight");
            b.ffn_down = g.leer_tensor(p + "ffn_down.weight");

            if ((i + 1) % 8 == 0 || i == num_capas - 1)
                std::cout << "  bloque " << (i+1) << "/" << num_capas << "\n";
        }

        // --- Vocabulario y tokenizador ---
        if (auto* v = g.arreglo_textos("tokenizer.ggml.tokens")) {
            tokens_texto = *v;
            std::cout << "Vocabulario: " << tokens_texto.size() << " tokens\n";

            // Construir el encoder BPE si el GGUF trae las reglas de fusion
            if (auto* m = g.arreglo_textos("tokenizer.ggml.merges")) {
                bpe.construir(tokens_texto, *m);
                std::cout << "Tokenizador BPE: " << m->size() << " reglas\n";
            } else {
                std::cout << "Aviso: sin reglas de merge, no habra encoder\n";
            }
        }

        // Tokens especiales
        bos = (int)g.entero("tokenizer.ggml.bos_token_id", -1);
        eos = (int)g.entero("tokenizer.ggml.eos_token_id", -1);

        std::cout << "Modelo cargado.\n";
    }

    // ------------------------------------------------------------------
    // Texto -> tokens
    // ------------------------------------------------------------------
    // Los tokens especiales como <|im_start|> deben reconocerse ANTES de
    // pasar por BPE; si no, el algoritmo los partiria en pedazos ("<", "|",
    // "im", "_start"...). Asi que buscamos esos marcadores primero y solo
    // tokenizamos el texto que hay entre ellos.
    std::vector<int> codificar(const std::string& texto) {
        static const std::vector<std::string> especiales = {
            "<|im_start|>", "<|im_end|>", "<|endoftext|>"
        };

        std::vector<int> ids;
        size_t i = 0;

        while (i < texto.size()) {
            // Buscar el proximo token especial
            size_t pos_min = std::string::npos;
            const std::string* cual = nullptr;
            for (const auto& esp : especiales) {
                size_t p = texto.find(esp, i);
                if (p != std::string::npos && p < pos_min) {
                    pos_min = p;
                    cual = &esp;
                }
            }

            if (cual == nullptr) {
                // No hay mas especiales: tokenizar el resto
                if (i < texto.size()) {
                    auto sub = bpe.codificar(texto.substr(i));
                    ids.insert(ids.end(), sub.begin(), sub.end());
                }
                break;
            }

            // Tokenizar lo que hay antes del especial
            if (pos_min > i) {
                auto sub = bpe.codificar(texto.substr(i, pos_min - i));
                ids.insert(ids.end(), sub.begin(), sub.end());
            }

            // Agregar el especial como un solo token
            int id = bpe.id_de(*cual);
            if (id >= 0) ids.push_back(id);

            i = pos_min + cual->size();
        }

        return ids;
    }

    // Arma un prompt en formato ChatML de Qwen:
    //   <|im_start|>system\n{sistema}<|im_end|>\n
    //   <|im_start|>user\n{usuario}<|im_end|>\n
    //   <|im_start|>assistant\n
    std::vector<int> armar_chat(const std::string& usuario,
                                const std::string& sistema = "") {
        std::string p;
        if (!sistema.empty())
            p += "<|im_start|>system\n" + sistema + "<|im_end|>\n";
        p += "<|im_start|>user\n" + usuario + "<|im_end|>\n";
        p += "<|im_start|>assistant\n";
        return codificar(p);
    }

    // ------------------------------------------------------------------
    // Feed-forward SwiGLU con matmul paralelo
    // ------------------------------------------------------------------
    Matrix feedforward(const Matrix& x, const BloqueQwen& b) {
        // Los pesos del GGUF vienen [salida x entrada], asi que usamos
        // la version con transpuesta implicita.
        Matrix gate = matmul_transpuesta_paralelo(x, b.ffn_gate);  // [1 x d_ff]
        Matrix up   = matmul_transpuesta_paralelo(x, b.ffn_up);    // [1 x d_ff]

        // La compuerta: SiLU(gate) * up
        for (size_t i = 0; i < gate.data.size(); i++)
            gate.data[i] = silu(gate.data[i]) * up.data[i];

        return matmul_transpuesta_paralelo(gate, b.ffn_down);      // [1 x d_model]
    }

    // ------------------------------------------------------------------
    // Un bloque transformer completo
    // ------------------------------------------------------------------
    Matrix bloque(const Matrix& x, const BloqueQwen& b,
                  int pos, CacheQwen& cache) {
        // Sub-bloque de atencion
        Matrix norm1 = rmsnorm(x, b.attn_norm, eps);
        Matrix attn = atencion_qwen(norm1,
                                    b.wq, b.bq, b.wk, b.bk, b.wv, b.bv, b.wo,
                                    n_heads, n_kv, pos, rope_base, cache);
        Matrix x1 = x.add(attn);

        // Sub-bloque de feed-forward
        Matrix norm2 = rmsnorm(x1, b.ffn_norm, eps);
        Matrix ff = feedforward(norm2, b);
        return x1.add(ff);
    }

    // ------------------------------------------------------------------
    // Forward pass de un token
    // ------------------------------------------------------------------
    //   token: ID del token
    //   pos:   su posicion en la secuencia (para RoPE)
    //   cache: se actualiza dentro
    //   devuelve: logits [1 x vocab]
    Matrix forward(int token, int pos, KVCacheQwen& cache) {
        if (token < 0 || token >= vocab)
            throw std::invalid_argument("Token fuera de rango: " +
                                        std::to_string(token));

        // 1. Embedding del token. Sin sumar posicion: RoPE se encarga
        //    de eso dentro de la atencion.
        Matrix x(1, d_model);
        for (int j = 0; j < d_model; j++)
            x.at(0, j) = token_embd.at(token, j);

        // 2. Los bloques
        for (int i = 0; i < num_capas; i++)
            x = bloque(x, bloques[i], pos, cache.capas[i]);

        // 3. RMSNorm final
        x = rmsnorm(x, output_norm, eps);

        // 4. Proyeccion a logits
        const Matrix& W = usa_weight_tying ? token_embd : output_w;
        return matmul_transpuesta_paralelo(x, W);
    }

    // Procesa una secuencia completa, devolviendo los logits del ultimo
    Matrix procesar_prompt(const std::vector<int>& tokens,
                           KVCacheQwen& cache) {
        cache.limpiar();
        Matrix logits;
        for (size_t i = 0; i < tokens.size(); i++)
            logits = forward(tokens[i], (int)i, cache);
        return logits;
    }

    // Crea un cache dimensionado para este modelo
    KVCacheQwen crear_cache(int max_tokens = 2048) {
        KVCacheQwen c;
        c.inicializar(num_capas, max_tokens, dim_kv());
        return c;
    }

    // Decodifica tokens a texto usando el vocabulario del GGUF.
    // Qwen usa el mismo esquema byte-level que GPT-2 ('Ġ' = espacio).
    std::string decodificar(const std::vector<int>& tokens) {
        std::string disfrazado;
        for (int id : tokens)
            if (id >= 0 && id < (int)tokens_texto.size())
                disfrazado += tokens_texto[id];
        return deshacer_byte_level(disfrazado);
    }

    std::string decodificar_uno(int token) {
        return decodificar({token});
    }

private:
    // Deshace el mapeo byte-level de GPT-2/Qwen.
    // Los tokens del vocabulario usan caracteres "disfrazados":
    // el espacio es 'Ġ' (U+0120), el salto de linea 'Ċ' (U+010A).
    std::string deshacer_byte_level(const std::string& s) {
        static std::map<int,int> decoder = construir_decoder();

        // Leer los codepoints del string UTF-8
        std::string salida;
        size_t i = 0;
        while (i < s.size()) {
            unsigned char c = s[i];
            int cp, len;
            if (c < 0x80)      { cp = c;        len = 1; }
            else if (c < 0xE0) { cp = c & 0x1F; len = 2; }
            else if (c < 0xF0) { cp = c & 0x0F; len = 3; }
            else               { cp = c & 0x07; len = 4; }
            for (int k = 1; k < len && i + k < s.size(); k++)
                cp = (cp << 6) | (s[i+k] & 0x3F);
            i += len;

            auto it = decoder.find(cp);
            salida += (char)(unsigned char)(it != decoder.end() ? it->second : cp);
        }
        return salida;
    }

    static std::map<int,int> construir_decoder() {
        std::vector<int> bs, cs;
        for (int i = (int)'!'; i <= (int)'~'; i++) bs.push_back(i);
        for (int i = 0xA1; i <= 0xAC; i++) bs.push_back(i);
        for (int i = 0xAE; i <= 0xFF; i++) bs.push_back(i);
        cs = bs;
        int n = 0;
        for (int b = 0; b < 256; b++) {
            if (std::find(bs.begin(), bs.end(), b) == bs.end()) {
                bs.push_back(b);
                cs.push_back(256 + n);
                n++;
            }
        }
        std::map<int,int> dec;
        for (size_t i = 0; i < bs.size(); i++) dec[cs[i]] = bs[i];
        return dec;
    }
};

#endif // QWEN_H
