// softmax.h
// Softmax por filas, con el truco de estabilidad numerica (restar el maximo).
// Es un ingrediente de attention. Lo aislamos y verificamos por separado
// porque un softmax mal hecho es la causa #1 de NaN en el transformer.

#ifndef SOFTMAX_H
#define SOFTMAX_H

#include "matrix.h"
#include <cmath>

// Aplica softmax a CADA fila de forma independiente.
// Cada fila de salida: valores positivos que suman 1.
//
// Por fila:
//   m = maximo de la fila            (para estabilidad)
//   e_j = exp(x_j - m)               (nunca exponenciamos un numero enorme)
//   salida_j = e_j / suma(e_j)
Matrix softmax(const Matrix& x) {
    int n = x.rows;
    int d = x.cols;
    Matrix salida(n, d);

    for (int i = 0; i < n; i++) {
        // 1. encontrar el maximo de la fila
        float m = x.at(i, 0);
        for (int j = 1; j < d; j++)
            if (x.at(i, j) > m) m = x.at(i, j);

        // 2. exponenciar (x - m) y acumular la suma
        float suma = 0.0f;
        for (int j = 0; j < d; j++) {
            float e = std::exp(x.at(i, j) - m);
            salida.at(i, j) = e;
            suma += e;
        }

        // 3. dividir cada valor entre la suma -> probabilidades que suman 1
        for (int j = 0; j < d; j++)
            salida.at(i, j) /= suma;
    }
    return salida;
}

#endif // SOFTMAX_H
