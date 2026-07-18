// layers.h
// Fin de semana 3: las primeras capas que PROCESAN datos.
//   - embeddings: convierte token IDs en la matriz de vectores del modelo
//   - layernorm:  normaliza cada fila (cada token) para estabilizar
//
// A partir de aqui empezamos a construir el flujo real del transformer.

#ifndef LAYERS_H
#define LAYERS_H

#include "matrix.h"
#include <vector>
#include <cmath>
#include <stdexcept>

// ----------------------------------------------------------------------
// EMBEDDINGS
// ----------------------------------------------------------------------
// Entrada: lista de token IDs, p.ej. {15496, 995}
// wte: token embeddings   [vocab x 768]   (fila = un token)
// wpe: positional embeddings [max_pos x 768] (fila = una posicion)
//
// Para cada token en la posicion p:
//   salida[p] = wte[token_id]  +  wpe[p]
//
// Resultado: matriz [num_tokens x 768]
Matrix embeddings(const std::vector<int>& tokens,
                  const Matrix& wte,
                  const Matrix& wpe) {
    int n = (int)tokens.size();
    int d = wte.cols;  // 768

    Matrix salida(n, d);
    for (int p = 0; p < n; p++) {
        int id = tokens[p];
        if (id < 0 || id >= wte.rows)
            throw std::invalid_argument("Token ID fuera de rango");
        if (p >= wpe.rows)
            throw std::invalid_argument("Posicion fuera del maximo de contexto");

        // sumar el vector del token y el vector de la posicion
        for (int j = 0; j < d; j++)
            salida.at(p, j) = wte.at(id, j) + wpe.at(p, j);
    }
    return salida;
}

// ----------------------------------------------------------------------
// LAYERNORM
// ----------------------------------------------------------------------
// Normaliza cada FILA de forma independiente.
// x:     [n x d]  la entrada
// gamma: [1 x d]  (ln.weight) escala aprendida
// beta:  [1 x d]  (ln.bias)   desplazamiento aprendido
//
// Por cada fila:
//   media = promedio de los d valores
//   var   = promedio de (valor - media)^2
//   norm  = (valor - media) / sqrt(var + eps)
//   salida = norm * gamma + beta
Matrix layernorm(const Matrix& x,
                 const Matrix& gamma,
                 const Matrix& beta,
                 float eps = 1e-5f) {
    int n = x.rows;
    int d = x.cols;

    if (gamma.cols != d || beta.cols != d)
        throw std::invalid_argument("layernorm: gamma/beta no coinciden con d");

    Matrix salida(n, d);
    for (int i = 0; i < n; i++) {
        // 1. media de la fila
        float media = 0.0f;
        for (int j = 0; j < d; j++) media += x.at(i, j);
        media /= d;

        // 2. varianza de la fila
        float var = 0.0f;
        for (int j = 0; j < d; j++) {
            float dif = x.at(i, j) - media;
            var += dif * dif;
        }
        var /= d;

        // 3. normalizar, escalar y desplazar
        float inv_std = 1.0f / std::sqrt(var + eps);
        for (int j = 0; j < d; j++) {
            float norm = (x.at(i, j) - media) * inv_std;
            salida.at(i, j) = norm * gamma.at(0, j) + beta.at(0, j);
        }
    }
    return salida;
}

#endif // LAYERS_H
