// Verifica que la atencion con cache da lo mismo que la atencion normal,
// usando pesos sinteticos (no necesita el modelo real).
#include "kvcache.h"
#include <iostream>
#include <cmath>

bool casi(float a, float b, float tol = 1e-4f) { return std::fabs(a-b) < tol; }

int main() {
    std::cout << "=== Test de equivalencia: con cache vs sin cache ===\n\n";
    
    int n = 4;          // 4 tokens
    int d = 8;          // d_model pequeno
    int cabezas = 2;    // 2 cabezas de 4
    
    // Entrada sintetica: 4 tokens de 8 dimensiones
    Matrix entrada(n, d);
    for (int i = 0; i < n*d; i++) entrada.data[i] = 0.01f * (i % 17) - 0.05f;
    
    // Pesos sinteticos
    Matrix c_attn_w(d, 3*d);
    for (int i = 0; i < d*3*d; i++) c_attn_w.data[i] = 0.003f * (i % 23) - 0.02f;
    Matrix c_attn_b(1, 3*d);
    for (int i = 0; i < 3*d; i++) c_attn_b.data[i] = 0.001f * i;
    Matrix c_proj_w(d, d);
    for (int i = 0; i < d*d; i++) c_proj_w.data[i] = 0.004f * (i % 13) - 0.01f;
    Matrix c_proj_b(1, d);
    for (int i = 0; i < d; i++) c_proj_b.data[i] = 0.002f * i;
    
    // --- Metodo 1: multi_head_attention normal, toda la secuencia ---
    Matrix salida_normal = multi_head_attention(
        entrada, c_attn_w, c_attn_b, c_proj_w, c_proj_b, cabezas);
    
    // --- Metodo 2: con cache, token por token ---
    CacheCapa cache;
    cache.reservar(100, d);
    
    Matrix salida_cache(n, d);
    for (int t = 0; t < n; t++) {
        // extraer el token t como matriz [1 x d]
        Matrix tok(1, d);
        for (int j = 0; j < d; j++) tok.at(0, j) = entrada.at(t, j);
        
        Matrix out = multi_head_cache(
            tok, c_attn_w, c_attn_b, c_proj_w, c_proj_b, cabezas, cache);
        
        for (int j = 0; j < d; j++) salida_cache.at(t, j) = out.at(0, j);
    }
    
    // --- Comparar ---
    bool ok = true;
    float max_dif = 0;
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < d; j++) {
            float dif = std::fabs(salida_normal.at(i,j) - salida_cache.at(i,j));
            if (dif > max_dif) max_dif = dif;
            if (!casi(salida_normal.at(i,j), salida_cache.at(i,j))) ok = false;
        }
    }
    
    std::cout << "Diferencia maxima: " << max_dif << "\n\n";
    if (ok) {
        std::cout << "[OK] La atencion con cache es matematicamente\n";
        std::cout << "     equivalente a la atencion normal.\n";
    } else {
        std::cout << "[FALLO] Las salidas difieren.\n";
        std::cout << "\nSalida normal (fila 0):\n  ";
        for (int j = 0; j < d; j++) std::cout << salida_normal.at(0,j) << " ";
        std::cout << "\nSalida cache (fila 0):\n  ";
        for (int j = 0; j < d; j++) std::cout << salida_cache.at(0,j) << " ";
        std::cout << "\n";
    }
    return ok ? 0 : 1;
}
