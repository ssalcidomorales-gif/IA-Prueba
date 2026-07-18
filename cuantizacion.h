// cuantizacion.h
// FASE 2 - Cuantizacion: convertir pesos comprimidos de vuelta a float32.
//
// EL PROBLEMA
// GPT-2 en float32 pesa 548 MB para 124M de parametros: 4 bytes por peso.
// Con esa proporcion, un modelo de 7B pesaria 28 GB. No cabe en RAM
// normal, mucho menos en los 8 GB de VRAM de una GPU de consumo.
//
// LA IDEA
// Los pesos de una red neuronal se agrupan en rangos estrechos. Si en un
// bloque de 32 pesos todos los valores estan entre -0.15 y 0.15, no hacen
// falta 32 bits de precision: guardas UNA escala para el bloque y luego
// cada peso como un entero pequeno relativo a esa escala.
//
//   peso real:       -0.0731
//   escala bloque:    0.0094   (fp16, una por bloque de 32)
//   valor cuantizado: -8       (4 bits)
//   reconstruido:     -8 * 0.0094 = -0.0752
//
// Se pierde precision, pero los modelos la toleran bien. Un 7B en Q4
// (4 GB) sigue siendo mucho mejor que un 3B en float32.

#ifndef CUANTIZACION_H
#define CUANTIZACION_H

#include <cstdint>
#include <cstring>
#include <vector>
#include <string>
#include <stdexcept>

// ----------------------------------------------------------------------
// TIPOS DE GGML
// ----------------------------------------------------------------------
// Los valores numericos son los que aparecen en el campo "type" de cada
// tensor dentro del archivo GGUF. No los inventamos: son el enum de ggml.
enum GGMLType : uint32_t {
    GGML_F32     = 0,
    GGML_F16     = 1,
    GGML_Q4_0    = 2,
    GGML_Q4_1    = 3,
    // 4 y 5 quedaron obsoletos (Q4_2, Q4_3)
    GGML_Q5_0    = 6,
    GGML_Q5_1    = 7,
    GGML_Q8_0    = 8,
    GGML_Q8_1    = 9,
    GGML_Q2_K    = 10,
    GGML_Q3_K    = 11,
    GGML_Q4_K    = 12,
    GGML_Q5_K    = 13,
    GGML_Q6_K    = 14,
    GGML_Q8_K    = 15,
};

inline const char* nombre_tipo(uint32_t t) {
    switch (t) {
        case GGML_F32:  return "F32";
        case GGML_F16:  return "F16";
        case GGML_Q4_0: return "Q4_0";
        case GGML_Q4_1: return "Q4_1";
        case GGML_Q5_0: return "Q5_0";
        case GGML_Q5_1: return "Q5_1";
        case GGML_Q8_0: return "Q8_0";
        case GGML_Q8_1: return "Q8_1";
        case GGML_Q2_K: return "Q2_K";
        case GGML_Q3_K: return "Q3_K";
        case GGML_Q4_K: return "Q4_K";
        case GGML_Q5_K: return "Q5_K";
        case GGML_Q6_K: return "Q6_K";
        case GGML_Q8_K: return "Q8_K";
        default:        return "DESCONOCIDO";
    }
}

// Cuantos valores hay en cada bloque de este tipo
inline int elementos_por_bloque(uint32_t tipo) {
    switch (tipo) {
        case GGML_F32:
        case GGML_F16:  return 1;
        case GGML_Q4_0:
        case GGML_Q4_1:
        case GGML_Q5_0:
        case GGML_Q5_1:
        case GGML_Q8_0:
        case GGML_Q8_1: return 32;
        // Los K-quants usan super-bloques de 256
        case GGML_Q2_K:
        case GGML_Q3_K:
        case GGML_Q4_K:
        case GGML_Q5_K:
        case GGML_Q6_K:
        case GGML_Q8_K: return 256;
        default: return 0;
    }
}

// Cuantos BYTES ocupa cada bloque
inline int bytes_por_bloque(uint32_t tipo) {
    switch (tipo) {
        case GGML_F32:  return 4;
        case GGML_F16:  return 2;
        // Q4_0: escala fp16 (2) + 32 valores a 4 bits (16) = 18
        case GGML_Q4_0: return 18;
        // Q4_1: escala + minimo (4) + 16 = 20
        case GGML_Q4_1: return 20;
        // Q5_0: escala (2) + bits altos (4) + nibbles (16) = 22
        case GGML_Q5_0: return 22;
        case GGML_Q5_1: return 24;
        // Q8_0: escala fp16 (2) + 32 int8 (32) = 34
        case GGML_Q8_0: return 34;
        case GGML_Q8_1: return 36;
        default: return 0;
    }
}

// ----------------------------------------------------------------------
// CONVERSION FP16 -> FP32
// ----------------------------------------------------------------------
// Las escalas se guardan en media precision (16 bits). Hay que
// desempaquetarlas a mano porque C++ no trae fp16 nativo.
//
// Formato IEEE 754 half:
//   bit 15    : signo
//   bits 14-10: exponente (5 bits, sesgo 15)
//   bits 9-0  : mantisa (10 bits)
inline float fp16_a_fp32(uint16_t h) {
    uint32_t signo     = (uint32_t)(h & 0x8000) << 16;
    uint32_t exponente = (h >> 10) & 0x1F;
    uint32_t mantisa   =  h & 0x03FF;

    uint32_t bits;
    if (exponente == 0) {
        if (mantisa == 0) {
            // Cero (con signo)
            bits = signo;
        } else {
            // Subnormal: normalizarlo desplazando hasta que el bit
            // implicito quede en su lugar
            exponente = 127 - 15 + 1;
            while ((mantisa & 0x0400) == 0) {
                mantisa <<= 1;
                exponente--;
            }
            mantisa &= 0x03FF;
            bits = signo | (exponente << 23) | (mantisa << 13);
        }
    } else if (exponente == 0x1F) {
        // Infinito o NaN
        bits = signo | 0x7F800000 | (mantisa << 13);
    } else {
        // Normal: reajustar el sesgo del exponente (15 -> 127)
        bits = signo | ((exponente - 15 + 127) << 23) | (mantisa << 13);
    }

    float f;
    std::memcpy(&f, &bits, sizeof(f));
    return f;
}

// ----------------------------------------------------------------------
// DEQUANTIZACION POR TIPO
// ----------------------------------------------------------------------

// Q4_0: 32 valores en 18 bytes.
//   bytes 0-1 : escala d (fp16)
//   bytes 2-17: 32 nibbles de 4 bits, dos por byte
//
// Los nibbles guardan valores 0..15 que representan -8..7, por eso se
// resta 8 al reconstruir. El empaquetado NO es secuencial: el nibble
// bajo del byte j da el valor j, y el nibble alto da el valor j+16.
// Ese entrelazado es intencional en ggml (facilita el SIMD).
inline void dequantizar_q4_0(const uint8_t* datos, int num_bloques,
                             float* salida) {
    for (int b = 0; b < num_bloques; b++) {
        const uint8_t* bloque = datos + (size_t)b * 18;

        uint16_t d_bits;
        std::memcpy(&d_bits, bloque, 2);
        float d = fp16_a_fp32(d_bits);

        const uint8_t* qs = bloque + 2;
        float* dest = salida + (size_t)b * 32;

        for (int j = 0; j < 16; j++) {
            uint8_t byte = qs[j];
            int bajo  = (byte & 0x0F) - 8;    // valor j
            int alto  = (byte >>   4) - 8;    // valor j+16
            dest[j]      = bajo * d;
            dest[j + 16] = alto * d;
        }
    }
}

// Q4_1: como Q4_0 pero con un minimo ademas de la escala.
//   bytes 0-1: escala d (fp16)
//   bytes 2-3: minimo m (fp16)
//   bytes 4-19: los nibbles
// Aqui los nibbles son 0..15 sin restar nada: valor = q*d + m
inline void dequantizar_q4_1(const uint8_t* datos, int num_bloques,
                             float* salida) {
    for (int b = 0; b < num_bloques; b++) {
        const uint8_t* bloque = datos + (size_t)b * 20;

        uint16_t d_bits, m_bits;
        std::memcpy(&d_bits, bloque, 2);
        std::memcpy(&m_bits, bloque + 2, 2);
        float d = fp16_a_fp32(d_bits);
        float m = fp16_a_fp32(m_bits);

        const uint8_t* qs = bloque + 4;
        float* dest = salida + (size_t)b * 32;

        for (int j = 0; j < 16; j++) {
            uint8_t byte = qs[j];
            dest[j]      = (byte & 0x0F) * d + m;
            dest[j + 16] = (byte >>   4) * d + m;
        }
    }
}

// Q8_0: 32 valores en 34 bytes.
//   bytes 0-1 : escala d (fp16)
//   bytes 2-33: 32 enteros con signo de 8 bits
// El mas simple de todos: valor = q * d
inline void dequantizar_q8_0(const uint8_t* datos, int num_bloques,
                             float* salida) {
    for (int b = 0; b < num_bloques; b++) {
        const uint8_t* bloque = datos + (size_t)b * 34;

        uint16_t d_bits;
        std::memcpy(&d_bits, bloque, 2);
        float d = fp16_a_fp32(d_bits);

        const int8_t* qs = (const int8_t*)(bloque + 2);
        float* dest = salida + (size_t)b * 32;

        for (int j = 0; j < 32; j++)
            dest[j] = qs[j] * d;
    }
}

// Q5_0: 32 valores en 22 bytes.
//   bytes 0-1 : escala d (fp16)
//   bytes 2-5 : qh, un uint32 con el 5o bit de cada valor
//   bytes 6-21: los 4 bits bajos, dos por byte
// El bit extra se saca de qh y se pega arriba del nibble.
inline void dequantizar_q5_0(const uint8_t* datos, int num_bloques,
                             float* salida) {
    for (int b = 0; b < num_bloques; b++) {
        const uint8_t* bloque = datos + (size_t)b * 22;

        uint16_t d_bits;
        std::memcpy(&d_bits, bloque, 2);
        float d = fp16_a_fp32(d_bits);

        uint32_t qh;
        std::memcpy(&qh, bloque + 2, 4);

        const uint8_t* qs = bloque + 6;
        float* dest = salida + (size_t)b * 32;

        for (int j = 0; j < 16; j++) {
            uint8_t byte = qs[j];
            // Recuperar el 5o bit de cada mitad
            int bit_bajo = (int)((qh >> j) & 1) << 4;
            int bit_alto = (int)((qh >> (j + 16)) & 1) << 4;
            int q_bajo = ((byte & 0x0F) | bit_bajo) - 16;
            int q_alto = ((byte >>   4) | bit_alto) - 16;
            dest[j]      = q_bajo * d;
            dest[j + 16] = q_alto * d;
        }
    }
}

// F16: conversion directa
inline void dequantizar_f16(const uint8_t* datos, int num_elementos,
                            float* salida) {
    const uint16_t* src = (const uint16_t*)datos;
    for (int i = 0; i < num_elementos; i++)
        salida[i] = fp16_a_fp32(src[i]);
}

// ----------------------------------------------------------------------
// DESPACHADOR
// ----------------------------------------------------------------------
// Convierte cualquier tensor soportado a un vector de float32.
//   datos:        puntero al inicio de los bytes del tensor
//   tipo:         GGMLType
//   num_elementos: cuantos valores tiene el tensor
inline std::vector<float> dequantizar(const uint8_t* datos, uint32_t tipo,
                                      size_t num_elementos) {
    std::vector<float> salida(num_elementos);

    switch (tipo) {
        case GGML_F32:
            std::memcpy(salida.data(), datos, num_elementos * 4);
            break;

        case GGML_F16:
            dequantizar_f16(datos, (int)num_elementos, salida.data());
            break;

        case GGML_Q4_0:
            dequantizar_q4_0(datos, (int)(num_elementos / 32), salida.data());
            break;

        case GGML_Q4_1:
            dequantizar_q4_1(datos, (int)(num_elementos / 32), salida.data());
            break;

        case GGML_Q5_0:
            dequantizar_q5_0(datos, (int)(num_elementos / 32), salida.data());
            break;

        case GGML_Q8_0:
            dequantizar_q8_0(datos, (int)(num_elementos / 32), salida.data());
            break;

        default:
            throw std::runtime_error(
                std::string("Tipo de cuantizacion no soportado: ") +
                nombre_tipo(tipo) + " (" + std::to_string(tipo) + ")");
    }

    return salida;
}

// Cuantos bytes ocupa en disco un tensor de este tipo y tamano
inline size_t bytes_del_tensor(uint32_t tipo, size_t num_elementos) {
    int elems = elementos_por_bloque(tipo);
    int bytes = bytes_por_bloque(tipo);
    if (elems == 0 || bytes == 0) return 0;
    return (num_elementos / elems) * bytes;
}

#endif // CUANTIZACION_H
