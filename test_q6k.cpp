// test_q6k.cpp
// Verifica la dequantizacion Q6_K construyendo super-bloques a mano.

#include "cuantizacion.h"
#include <iostream>
#include <iomanip>
#include <cmath>
#include <cstring>
#include <vector>

int ok = 0, total = 0;
void afirmar(const std::string& n, bool c) {
    total++;
    if (c) { std::cout << "[OK]    " << n << "\n"; ok++; }
    else   { std::cout << "[FALLO] " << n << "\n"; }
}
bool casi(float a, float b, float tol=1e-3f){ return std::fabs(a-b)<tol; }

uint16_t fp32_a_fp16(float f) {
    uint32_t bits; std::memcpy(&bits,&f,4);
    uint32_t s=(bits>>16)&0x8000; int32_t e=((bits>>23)&0xFF)-127+15;
    uint32_t m=(bits>>13)&0x3FF;
    if(e<=0) return (uint16_t)s;
    if(e>=31) return (uint16_t)(s|0x7C00);
    return (uint16_t)(s|(e<<10)|m);
}

int main() {
    std::cout << "=== Tests de Q6_K ===\n\n";

    afirmar("Q6_K: 256 elementos por bloque",
            elementos_por_bloque(GGML_Q6_K) == 256);
    afirmar("Q6_K: 210 bytes por bloque",
            bytes_por_bloque(GGML_Q6_K) == 210);
    afirmar("bytes_del_tensor Q6_K(1024) = 840",
            bytes_del_tensor(GGML_Q6_K, 1024) == 840);

    // --- Bloque con todo en cero ---
    // Los nibbles a 0 representan -32 tras centrar. Con escala 0,
    // todo deberia salir 0.
    {
        std::vector<uint8_t> b(210, 0);
        uint16_t d = fp32_a_fp16(1.0f);
        std::memcpy(b.data()+208, &d, 2);
        // escalas en 0
        for (int i = 0; i < 16; i++) ((int8_t*)(b.data()+192))[i] = 0;

        std::vector<float> out(256);
        dequantizar_q6_k(b.data(), 1, out.data());

        bool todo_cero = true;
        for (float v : out) if (!casi(v, 0.0f)) todo_cero = false;
        afirmar("escalas en cero -> todo cero", todo_cero);
    }

    // --- Verificar el valor central: nibble 0 + bits altos 2 = 32 -> 0 ---
    {
        std::vector<uint8_t> b(210, 0);
        uint16_t d = fp32_a_fp16(1.0f);
        std::memcpy(b.data()+208, &d, 2);
        for (int i = 0; i < 16; i++) ((int8_t*)(b.data()+192))[i] = 1;

        // Para que el valor 0 (posicion 0) sea 32 (que centrado da 0):
        // nibble bajo de ql[0] = 0, bits 0-1 de qh[0] = 2
        // porque 0 | (2 << 4) = 32
        b[0] = 0x00;        // ql[0]: nibble bajo 0
        b[128] = 0x02;      // qh[0]: bits 0-1 = 2

        std::vector<float> out(256);
        dequantizar_q6_k(b.data(), 1, out.data());

        std::cout << "        valor[0] con q=32 (centrado=0): " << out[0] << "\n";
        afirmar("q=32 se centra a 0", casi(out[0], 0.0f));
    }

    // --- Un valor concreto: q = 63 (maximo) -> centrado 31 ---
    {
        std::vector<uint8_t> b(210, 0);
        uint16_t d = fp32_a_fp16(0.5f);
        std::memcpy(b.data()+208, &d, 2);
        for (int i = 0; i < 16; i++) ((int8_t*)(b.data()+192))[i] = 2;

        // valor maximo: nibble = 15, bits altos = 3 -> 15 | (3<<4) = 63
        b[0] = 0x0F;
        b[128] = 0x03;

        std::vector<float> out(256);
        dequantizar_q6_k(b.data(), 1, out.data());

        // esperado: d * escala * (63-32) = 0.5 * 2 * 31 = 31
        std::cout << "        valor[0] con q=63: " << out[0]
                  << " (esperado 31)\n";
        afirmar("q=63 -> d*escala*31", casi(out[0], 31.0f, 0.1f));
    }

    // --- El valor minimo: q = 0 -> centrado -32 ---
    {
        std::vector<uint8_t> b(210, 0);
        uint16_t d = fp32_a_fp16(1.0f);
        std::memcpy(b.data()+208, &d, 2);
        for (int i = 0; i < 16; i++) ((int8_t*)(b.data()+192))[i] = 1;
        // ql[0] = 0 y qh[0] = 0 -> q = 0 -> centrado = -32
        std::vector<float> out(256);
        dequantizar_q6_k(b.data(), 1, out.data());
        std::cout << "        valor[0] con q=0: " << out[0]
                  << " (esperado -32)\n";
        afirmar("q=0 -> -32", casi(out[0], -32.0f, 0.1f));
    }

    // --- Las escalas por sub-bloque se aplican independientemente ---
    {
        std::vector<uint8_t> b(210, 0);
        uint16_t d = fp32_a_fp16(1.0f);
        std::memcpy(b.data()+208, &d, 2);
        // escala 1 para el primer sub-bloque, 2 para el resto
        int8_t* sc = (int8_t*)(b.data()+192);
        sc[0] = 1;
        for (int i = 1; i < 16; i++) sc[i] = 2;

        // todos los quants en 0 -> centrado -32
        std::vector<float> out(256);
        dequantizar_q6_k(b.data(), 1, out.data());

        // posicion 0 usa sc[0]=1 -> -32
        // posicion 16 usa sc[1]=2 -> -64
        std::cout << "        valor[0] (escala 1): " << out[0] << "\n";
        std::cout << "        valor[16] (escala 2): " << out[16] << "\n";
        afirmar("escalas por sub-bloque aplicadas",
                casi(out[0], -32.0f, 0.1f) && casi(out[16], -64.0f, 0.1f));
    }

    // --- El despachador reconoce Q6_K ---
    {
        std::vector<uint8_t> b(210, 0);
        uint16_t d = fp32_a_fp16(1.0f);
        std::memcpy(b.data()+208, &d, 2);
        for (int i = 0; i < 16; i++) ((int8_t*)(b.data()+192))[i] = 1;

        bool sin_error = true;
        try {
            auto v = dequantizar(b.data(), GGML_Q6_K, 256);
            if (v.size() != 256) sin_error = false;
        } catch (...) { sin_error = false; }
        afirmar("el despachador maneja Q6_K", sin_error);
    }

    std::cout << "\n=== " << ok << "/" << total << " pruebas pasaron ===\n";
    return (ok==total)?0:1;
}
