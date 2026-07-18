// test_cuantizacion.cpp
// Verifica la dequantizacion construyendo bloques con valores conocidos
// y comprobando que se reconstruyen bien. No necesita ningun modelo.

#include "cuantizacion.h"
#include <iostream>
#include <iomanip>
#include <cmath>
#include <cstring>

int ok = 0, total = 0;

void afirmar(const std::string& nombre, bool cond) {
    total++;
    if (cond) { std::cout << "[OK]    " << nombre << "\n"; ok++; }
    else      { std::cout << "[FALLO] " << nombre << "\n"; }
}

bool casi(float a, float b, float tol = 1e-4f) { return std::fabs(a-b) < tol; }

// Empaqueta un float en fp16 (para construir casos de prueba)
uint16_t fp32_a_fp16(float f) {
    uint32_t bits;
    std::memcpy(&bits, &f, 4);
    uint32_t signo = (bits >> 16) & 0x8000;
    int32_t  exp   = ((bits >> 23) & 0xFF) - 127 + 15;
    uint32_t mant  = (bits >> 13) & 0x3FF;
    if (exp <= 0) return (uint16_t)signo;
    if (exp >= 31) return (uint16_t)(signo | 0x7C00);
    return (uint16_t)(signo | (exp << 10) | mant);
}

int main() {
    std::cout << "=== Tests de dequantizacion ===\n\n";

    // --- Test 1: fp16 -> fp32 con valores conocidos ---
    {
        // 0x3C00 = 1.0, 0x4000 = 2.0, 0xBC00 = -1.0, 0x0000 = 0.0
        afirmar("fp16 0x3C00 = 1.0",  casi(fp16_a_fp32(0x3C00), 1.0f));
        afirmar("fp16 0x4000 = 2.0",  casi(fp16_a_fp32(0x4000), 2.0f));
        afirmar("fp16 0xBC00 = -1.0", casi(fp16_a_fp32(0xBC00), -1.0f));
        afirmar("fp16 0x0000 = 0.0",  casi(fp16_a_fp32(0x0000), 0.0f));
        // 0x3555 ~ 0.3333
        afirmar("fp16 0x3555 ~ 0.333", casi(fp16_a_fp32(0x3555), 0.33325f, 1e-3f));
    }

    // --- Test 2: tamanos de bloque correctos ---
    {
        afirmar("Q4_0: 32 elems, 18 bytes",
                elementos_por_bloque(GGML_Q4_0)==32 && bytes_por_bloque(GGML_Q4_0)==18);
        afirmar("Q8_0: 32 elems, 34 bytes",
                elementos_por_bloque(GGML_Q8_0)==32 && bytes_por_bloque(GGML_Q8_0)==34);
        afirmar("Q5_0: 32 elems, 22 bytes",
                elementos_por_bloque(GGML_Q5_0)==32 && bytes_por_bloque(GGML_Q5_0)==22);
    }

    // --- Test 3: Q8_0 con un bloque construido a mano ---
    // escala = 0.5, valores 0,1,2,...,31 -> reconstruido = 0, 0.5, 1.0, ...
    {
        uint8_t bloque[34];
        uint16_t d = fp32_a_fp16(0.5f);
        std::memcpy(bloque, &d, 2);
        for (int j = 0; j < 32; j++) ((int8_t*)(bloque+2))[j] = (int8_t)j;

        float salida[32];
        dequantizar_q8_0(bloque, 1, salida);

        bool bien = true;
        for (int j = 0; j < 32; j++)
            if (!casi(salida[j], j * 0.5f)) bien = false;
        afirmar("Q8_0: valor = q * escala", bien);
    }

    // --- Test 4: Q8_0 con valores negativos ---
    {
        uint8_t bloque[34];
        uint16_t d = fp32_a_fp16(0.25f);
        std::memcpy(bloque, &d, 2);
        int8_t vals[32] = {-128, -64, -1, 0, 1, 64, 127};
        for (int j = 0; j < 32; j++) ((int8_t*)(bloque+2))[j] = (j < 7) ? vals[j] : 0;

        float salida[32];
        dequantizar_q8_0(bloque, 1, salida);
        afirmar("Q8_0: maneja negativos",
                casi(salida[0], -32.0f) && casi(salida[2], -0.25f) &&
                casi(salida[6], 31.75f));
    }

    // --- Test 5: Q4_0, el empaquetado entrelazado ---
    // Cada byte tiene dos nibbles: el bajo es el valor j, el alto el j+16.
    // Los nibbles 0..15 representan -8..7 (se resta 8).
    {
        uint8_t bloque[18];
        uint16_t d = fp32_a_fp16(1.0f);   // escala 1 para leer directo
        std::memcpy(bloque, &d, 2);
        // byte 0: nibble bajo = 8 (-> 0), nibble alto = 9 (-> 1)
        bloque[2] = 0x98;
        // byte 1: nibble bajo = 0 (-> -8), nibble alto = 15 (-> 7)
        bloque[3] = 0xF0;
        for (int j = 2; j < 16; j++) bloque[2+j] = 0x88;  // ceros

        float salida[32];
        dequantizar_q4_0(bloque, 1, salida);

        afirmar("Q4_0: nibble bajo -> posicion j",
                casi(salida[0], 0.0f) && casi(salida[1], -8.0f));
        afirmar("Q4_0: nibble alto -> posicion j+16",
                casi(salida[16], 1.0f) && casi(salida[17], 7.0f));
    }

    // --- Test 6: Q4_0 con escala real ---
    {
        uint8_t bloque[18];
        float escala = 0.01f;
        uint16_t d = fp32_a_fp16(escala);
        float escala_real = fp16_a_fp32(d);  // lo que de verdad cabe en fp16
        std::memcpy(bloque, &d, 2);
        bloque[2] = 0x0F;  // bajo=15 (->7), alto=0 (->-8)
        for (int j = 1; j < 16; j++) bloque[2+j] = 0x88;

        float salida[32];
        dequantizar_q4_0(bloque, 1, salida);
        afirmar("Q4_0: aplica la escala",
                casi(salida[0], 7 * escala_real, 1e-5f) &&
                casi(salida[16], -8 * escala_real, 1e-5f));
    }

    // --- Test 7: el despachador general ---
    {
        uint8_t bloque[34];
        uint16_t d = fp32_a_fp16(2.0f);
        std::memcpy(bloque, &d, 2);
        for (int j = 0; j < 32; j++) ((int8_t*)(bloque+2))[j] = 3;

        auto v = dequantizar(bloque, GGML_Q8_0, 32);
        bool bien = (v.size() == 32);
        for (float x : v) if (!casi(x, 6.0f)) bien = false;
        afirmar("despachador: Q8_0 correcto", bien);
    }

    // --- Test 8: calculo de tamano en disco ---
    {
        // 1024 elementos en Q4_0 = 32 bloques * 18 bytes = 576
        afirmar("bytes_del_tensor Q4_0(1024) = 576",
                bytes_del_tensor(GGML_Q4_0, 1024) == 576);
        // 1024 en Q8_0 = 32 * 34 = 1088
        afirmar("bytes_del_tensor Q8_0(1024) = 1088",
                bytes_del_tensor(GGML_Q8_0, 1024) == 1088);
        // 1024 en F32 = 4096
        afirmar("bytes_del_tensor F32(1024) = 4096",
                bytes_del_tensor(GGML_F32, 1024) == 4096);
    }

    // --- Test 9: tipo no soportado lanza excepcion ---
    {
        uint8_t basura[64] = {0};
        bool lanzo = false;
        try { dequantizar(basura, GGML_Q4_K, 256); }
        catch (const std::exception&) { lanzo = true; }
        afirmar("tipo no soportado lanza error", lanzo);
    }

    // --- Demostracion: cuanto se ahorra ---
    std::cout << "\n=== Compresion por formato (1000M de parametros) ===\n";
    std::cout << std::fixed << std::setprecision(2);
    struct { uint32_t t; const char* n; } formatos[] = {
        {GGML_F32, "F32"}, {GGML_F16, "F16"},
        {GGML_Q8_0, "Q8_0"}, {GGML_Q5_0, "Q5_0"}, {GGML_Q4_0, "Q4_0"}
    };
    for (auto& f : formatos) {
        double bytes = (double)bytes_del_tensor(f.t, 1000000000ULL);
        double gb = bytes / (1024.0*1024*1024);
        double bits = bytes * 8.0 / 1e9;
        std::cout << "  " << std::setw(5) << f.n << ": "
                  << std::setw(6) << gb << " GB  ("
                  << bits << " bits por peso)\n";
    }

    std::cout << "\n=== " << ok << "/" << total << " pruebas pasaron ===\n";
    return (ok == total) ? 0 : 1;
}
