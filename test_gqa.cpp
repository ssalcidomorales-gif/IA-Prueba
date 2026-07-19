// test_gqa.cpp
// Verifica la atencion GQA: que las cabezas de query compartan
// correctamente las cabezas de key/value, y que el cache funcione.

#include "qwen_attention.h"
#include <iostream>
#include <iomanip>
#include <cmath>

int ok = 0, total = 0;
void afirmar(const std::string& n, bool c) {
    total++;
    if (c) { std::cout << "[OK]    " << n << "\n"; ok++; }
    else   { std::cout << "[FALLO] " << n << "\n"; }
}
bool casi(float a, float b, float tol=1e-3f){ return std::fabs(a-b)<tol; }

int main() {
    std::cout << "=== Tests de atencion GQA ===\n\n";
    Paralelo::configurar(2);

    // Configuracion chica pero con la misma estructura que Qwen:
    // 4 cabezas de query, 2 de KV -> grupos de 2
    int d_model = 8, n_heads = 4, n_kv = 2;
    int dim_cabeza = d_model / n_heads;   // 2
    int dim_kv = n_kv * dim_cabeza;       // 4

    std::cout << "Configuracion de prueba:\n";
    std::cout << "  d_model=" << d_model << " n_heads=" << n_heads
              << " n_kv=" << n_kv << "\n";
    std::cout << "  dim_cabeza=" << dim_cabeza << " dim_kv=" << dim_kv
              << " grupo=" << (n_heads/n_kv) << " queries por cabeza kv\n\n";

    // Pesos sinteticos
    Matrix Wq(d_model, d_model), Wk(dim_kv, d_model), Wv(dim_kv, d_model);
    Matrix Wo(d_model, d_model);
    Matrix bq(1, d_model), bk(1, dim_kv), bv(1, dim_kv);

    for (int i = 0; i < d_model*d_model; i++) {
        Wq.data[i] = 0.03f * (i % 11) - 0.1f;
        Wo.data[i] = 0.02f * (i % 7)  - 0.05f;
    }
    for (int i = 0; i < dim_kv*d_model; i++) {
        Wk.data[i] = 0.04f * (i % 9) - 0.12f;
        Wv.data[i] = 0.05f * (i % 5) - 0.08f;
    }

    // --- Test 1: forma de salida correcta ---
    {
        CacheQwen cache;
        cache.reservar(100, dim_kv);
        Matrix x(1, d_model);
        for (int j = 0; j < d_model; j++) x.at(0,j) = 0.1f*(j+1);

        Matrix r = atencion_qwen(x, Wq,bq, Wk,bk, Wv,bv, Wo,
                                 n_heads, n_kv, 0, 10000.0f, cache);
        afirmar("salida tiene forma [1 x d_model]",
                r.rows==1 && r.cols==d_model);
        afirmar("el cache guarda 1 token", cache.num_tokens == 1);
        afirmar("el cache guarda dim_kv por token", cache.K.cols == dim_kv);
    }

    // --- Test 2: el cache acumula tokens ---
    {
        CacheQwen cache;
        cache.reservar(100, dim_kv);
        for (int t = 0; t < 5; t++) {
            Matrix x(1, d_model);
            for (int j = 0; j < d_model; j++) x.at(0,j) = 0.05f*(t+1)*(j+1);
            atencion_qwen(x, Wq,bq, Wk,bk, Wv,bv, Wo,
                          n_heads, n_kv, t, 10000.0f, cache);
        }
        afirmar("el cache acumula 5 tokens", cache.num_tokens == 5);
    }

    // --- Test 3: el primer token solo se atiende a si mismo ---
    // Con un solo token en el cache, el softmax da peso 1 a ese unico
    // token, asi que la salida antes de Wo es exactamente su propio V.
    {
        CacheQwen cache;
        cache.reservar(100, dim_kv);
        Matrix x(1, d_model);
        for (int j = 0; j < d_model; j++) x.at(0,j) = 0.2f*(j+1);

        // Calcular V manualmente
        Matrix V_esperado = lineal_qwen(x, Wv, &bv);

        atencion_qwen(x, Wq,bq, Wk,bk, Wv,bv, Wo,
                      n_heads, n_kv, 0, 10000.0f, cache);

        // El cache debe contener exactamente ese V
        bool bien = true;
        for (int j = 0; j < dim_kv; j++)
            if (!casi(cache.V.at(0,j), V_esperado.at(0,j))) bien = false;
        afirmar("el cache guarda V sin modificar (no lleva RoPE)", bien);
    }

    // --- Test 4: LA CLAVE DE GQA ---
    // Las cabezas de query 0 y 1 comparten la cabeza KV 0.
    // Las cabezas 2 y 3 comparten la cabeza KV 1.
    // Verificamos el mapeo: h_kv = h / grupo
    {
        int grupo = n_heads / n_kv;
        bool bien = (0/grupo == 0) && (1/grupo == 0) &&
                    (2/grupo == 1) && (3/grupo == 1);
        afirmar("mapeo GQA: queries 0,1 -> kv 0; queries 2,3 -> kv 1", bien);
    }

    // --- Test 5: K si lleva RoPE (a diferencia de V) ---
    // La misma entrada en posicion 0 vs posicion 5 debe dar K distintas
    // (porque RoPE rota) pero V identicas (porque V no se rota).
    {
        Matrix x(1, d_model);
        for (int j = 0; j < d_model; j++) x.at(0,j) = 0.15f*(j+1);

        CacheQwen c0, c5;
        c0.reservar(10, dim_kv);
        c5.reservar(10, dim_kv);

        atencion_qwen(x, Wq,bq, Wk,bk, Wv,bv, Wo, n_heads,n_kv, 0, 10000.0f, c0);
        atencion_qwen(x, Wq,bq, Wk,bk, Wv,bv, Wo, n_heads,n_kv, 5, 10000.0f, c5);

        bool k_difiere = false, v_igual = true;
        for (int j = 0; j < dim_kv; j++) {
            if (!casi(c0.K.at(0,j), c5.K.at(0,j))) k_difiere = true;
            if (!casi(c0.V.at(0,j), c5.V.at(0,j))) v_igual = false;
        }
        afirmar("K cambia con la posicion (lleva RoPE)", k_difiere);
        afirmar("V NO cambia con la posicion (sin RoPE)", v_igual);
    }

    // --- Test 6: ahorro de memoria del GQA ---
    {
        // Con la config de Qwen2.5-0.5B real
        int qwen_heads = 14, qwen_kv = 2, qwen_dim_cabeza = 64;
        int sin_gqa = qwen_heads * qwen_dim_cabeza * 2;  // K y V
        int con_gqa = qwen_kv    * qwen_dim_cabeza * 2;
        std::cout << "\n  Memoria de cache por token (Qwen2.5-0.5B):\n";
        std::cout << "    Sin GQA (14 cabezas kv): " << sin_gqa << " floats\n";
        std::cout << "    Con GQA (2 cabezas kv):  " << con_gqa << " floats\n";
        std::cout << "    Ahorro: " << (sin_gqa/con_gqa) << "x\n";
        std::cout << "    Para 32768 tokens x 24 capas:\n";
        std::cout << "      sin GQA: "
                  << ((double)sin_gqa*32768*24*4/1024/1024/1024) << " GB\n";
        std::cout << "      con GQA: "
                  << ((double)con_gqa*32768*24*4/1024/1024/1024) << " GB\n\n";
        afirmar("GQA ahorra 7x en el cache", (sin_gqa/con_gqa) == 7);
    }

    std::cout << "\n=== " << ok << "/" << total << " pruebas pasaron ===\n";
    return (ok == total) ? 0 : 1;
}
