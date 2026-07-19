// qwen_ops.h
// FASE 3 - Las operaciones que Qwen2.5 usa y GPT-2 no.
//
//   RMSNorm  - normalizacion sin restar la media
//   RoPE     - posicion codificada rotando los vectores, no sumando
//   SwiGLU   - feed-forward con compuerta
//
// Cada una reemplaza a su equivalente de GPT-2:
//   LayerNorm      -> RMSNorm
//   wpe (posicion) -> RoPE
//   GELU + c_fc    -> SwiGLU con gate/up/down

#ifndef QWEN_OPS_H
#define QWEN_OPS_H

#include "matrix.h"
#include <cmath>
#include <vector>
#include <stdexcept>

// ----------------------------------------------------------------------
// RMSNorm
// ----------------------------------------------------------------------
// LayerNorm resta la media y divide por la desviacion estandar.
// RMSNorm se salta la resta: solo divide por la raiz cuadratica media.
//
//   LayerNorm: (x - media) / sqrt(varianza + eps) * gamma + beta
//   RMSNorm:    x / sqrt(promedio(x^2) + eps) * gamma
//
// Dos simplificaciones: no calcula la media, y no tiene beta (solo gamma).
// Resulta que restar la media aportaba poco y costaba una pasada extra
// sobre los datos, asi que los modelos modernos la quitaron.
//
//   x:     [n x d]
//   gamma: [1 x d]  (el tensor *_norm.weight)
Matrix rmsnorm(const Matrix& x, const Matrix& gamma, float eps = 1e-6f) {
    int n = x.rows;
    int d = x.cols;

    if (gamma.cols != d)
        throw std::invalid_argument("rmsnorm: gamma no coincide con d");

    Matrix salida(n, d);
    for (int i = 0; i < n; i++) {
        // promedio de los cuadrados de la fila
        float suma_cuad = 0.0f;
        for (int j = 0; j < d; j++) {
            float v = x.at(i, j);
            suma_cuad += v * v;
        }
        float rms_inv = 1.0f / std::sqrt(suma_cuad / d + eps);

        for (int j = 0; j < d; j++)
            salida.at(i, j) = x.at(i, j) * rms_inv * gamma.at(0, j);
    }
    return salida;
}

// ----------------------------------------------------------------------
// RoPE (Rotary Position Embedding)
// ----------------------------------------------------------------------
// GPT-2 sumaba un vector de posicion al embedding. Qwen no suma nada:
// ROTA los vectores Q y K segun su posicion.
//
// LA IDEA
// Toma cada par de dimensiones (2j, 2j+1) como un punto en un plano y
// girálo un angulo theta que depende de la posicion del token y del par:
//
//   theta = posicion / (base ^ (2j/d))
//
//   x'[2j]   = x[2j] * cos(theta) - x[2j+1] * sin(theta)
//   x'[2j+1] = x[2j] * sin(theta) + x[2j+1] * cos(theta)
//
// POR QUE FUNCIONA
// Cuando haces el producto punto entre una query en posicion m y una key
// en posicion n, las rotaciones se combinan y el resultado depende solo
// de la DIFERENCIA (m - n). La atencion percibe distancias relativas de
// forma natural, sin tabla de posiciones y sin limite duro de contexto.
//
// EL LAYOUT DE LOS PARES: hay DOS convenciones y elegir mal produce
// basura con caracteres raros (sintoma clasico).
//
//   Intercalada (GPT-J):  pares (0,1), (2,3), (4,5), ...
//   NeoX (mitad partida): pares (0, d/2), (1, d/2+1), (2, d/2+2), ...
//
// llama.cpp mapea Qwen a NEOX, asi que usamos la de mitad partida:
// la dimension j se empareja con la j + d/2.
//
//   x:        [n x dim_cabeza]  las filas son tokens
//   pos_ini:  posicion del primer token de x en la secuencia completa
//   base:     rope.freq_base del modelo (1000000 en Qwen2.5)
void aplicar_rope(Matrix& x, int pos_ini, float base = 1000000.0f) {
    int n = x.rows;
    int d = x.cols;
    int mitad = d / 2;

    for (int i = 0; i < n; i++) {
        int pos = pos_ini + i;
        for (int j = 0; j < mitad; j++) {
            // frecuencia de este par de dimensiones
            float exponente = (float)(2 * j) / (float)d;
            float theta = (float)pos / std::pow(base, exponente);
            float c = std::cos(theta);
            float s = std::sin(theta);

            // NeoX: se empareja j con j + mitad
            int a = j;
            int b = j + mitad;
            float xa = x.at(i, a);
            float xb = x.at(i, b);

            x.at(i, a) = xa * c - xb * s;
            x.at(i, b) = xa * s + xb * c;
        }
    }
}

// Version para un solo vector (caso del KV cache: un token a la vez).
// Misma convencion NeoX.
void aplicar_rope_fila(float* v, int d, int pos, float base = 1000000.0f) {
    int mitad = d / 2;
    for (int j = 0; j < mitad; j++) {
        float exponente = (float)(2 * j) / (float)d;
        float theta = (float)pos / std::pow(base, exponente);
        float c = std::cos(theta);
        float s = std::sin(theta);

        int a = j;
        int b = j + mitad;
        float va = v[a];
        float vb = v[b];
        v[a] = va * c - vb * s;
        v[b] = va * s + vb * c;
    }
}

// ----------------------------------------------------------------------
// SiLU (Sigmoid Linear Unit), tambien llamada Swish
// ----------------------------------------------------------------------
//   SiLU(x) = x * sigmoid(x) = x / (1 + e^-x)
//
// Es la activacion que usa SwiGLU. Se parece a GELU en forma pero es
// mas barata de calcular.
inline float silu(float x) {
    return x / (1.0f + std::exp(-x));
}

Matrix aplicar_silu(const Matrix& x) {
    Matrix salida(x.rows, x.cols);
    for (size_t i = 0; i < x.data.size(); i++)
        salida.data[i] = silu(x.data[i]);
    return salida;
}

// ----------------------------------------------------------------------
// SwiGLU: el feed-forward de Qwen
// ----------------------------------------------------------------------
// GPT-2 hacia:  x -> c_fc -> GELU -> c_proj
// Qwen hace:    x -> (gate, up) en paralelo, SiLU sobre gate,
//               multiplicar elemento a elemento, luego down
//
//   SwiGLU(x) = down( SiLU(x * W_gate) * (x * W_up) )
//
// LA INTUICION DE LA COMPUERTA
// El camino "gate" pasa por SiLU y produce valores entre ~0 y el valor
// original. Al multiplicarlo por el camino "up", actua como una valvula:
// decide cuanta senal de "up" deja pasar en cada dimension. Por eso se
// llama compuerta (gate).
//
//   entrada: [n x d]
//   W_gate:  [d x d_ff]
//   W_up:    [d x d_ff]
//   W_down:  [d_ff x d]
//   salida:  [n x d]
//
// Nota: esta version usa matmul simple. La version del modelo real
// usara matmul_paralelo (ver qwen.h).
Matrix swiglu(const Matrix& entrada,
    const Matrix& W_gate,
    const Matrix& W_up,
    const Matrix& W_down) {
    // Los dos caminos en paralelo
    Matrix gate = entrada.matmul(W_gate);   // [n x d_ff]
    Matrix up = entrada.matmul(W_up);     // [n x d_ff]

    // SiLU sobre el gate, y multiplicar elemento a elemento
    for (size_t i = 0; i < gate.data.size(); i++)
        gate.data[i] = silu(gate.data[i]) * up.data[i];

    // Proyectar de vuelta a la dimension del modelo
    return gate.matmul(W_down);             // [n x d]
}

#endif // QWEN_OPS_H