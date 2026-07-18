// matrix.h
// Fin de semana 1 - Los ladrillos del transformer.
// Una matriz es solo un bloque de floats en memoria, guardado fila por fila
// (row-major). Guardamos filas, columnas y los datos en un vector plano.
//
// Row-major significa: el elemento (fila f, columna c) esta en data[f * cols + c].
// Es la MISMA convencion que usa PyTorch, asi que cuando cargues los pesos
// de GPT-2 en la Fase 2, van a encajar sin transponer nada. Recuerdalo.

#ifndef MATRIX_H
#define MATRIX_H

#include <vector>
#include <stdexcept>
#include <cmath>
#include <iostream>
#include <iomanip>
#include <string>

class Matrix {
public:
    int rows;
    int cols;
    std::vector<float> data;  // tamano = rows * cols, guardado row-major

    // --- Constructores ---

    // Matriz vacia (0x0)
    Matrix() : rows(0), cols(0) {}

    // Matriz de rows x cols llena de ceros
    Matrix(int r, int c) : rows(r), cols(c), data(r * c, 0.0f) {}

    // Matriz a partir de una lista de listas (util para tests escritos a mano)
    Matrix(std::initializer_list<std::initializer_list<float>> lista) {
        rows = (int)lista.size();
        cols = rows > 0 ? (int)lista.begin()->size() : 0;
        data.reserve(rows * cols);
        for (const auto& fila : lista) {
            if ((int)fila.size() != cols)
                throw std::invalid_argument("Todas las filas deben tener el mismo largo");
            for (float v : fila) data.push_back(v);
        }
    }

    // --- Acceso a elementos ---
    // at(f, c) te da una referencia al elemento, para leer o escribir.
    // Ejemplo: m.at(0, 1) = 5.0f;  o  float x = m.at(2, 3);

    float& at(int f, int c) {
        return data[f * cols + c];
    }
    float at(int f, int c) const {
        return data[f * cols + c];
    }

    // --- Operaciones ---

    // Multiplicacion de matrices: (this) * (otra)
    // Si this es [m x n] y otra es [n x p], el resultado es [m x p].
    // Esta es LA operacion. El 90% del tiempo de computo de GPT-2 vive aqui.
    Matrix matmul(const Matrix& otra) const {
        if (cols != otra.rows)
            throw std::invalid_argument(
                "matmul: columnas de A (" + std::to_string(cols) +
                ") deben igualar filas de B (" + std::to_string(otra.rows) + ")");

        Matrix resultado(rows, otra.cols);
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < otra.cols; j++) {
                float suma = 0.0f;
                for (int k = 0; k < cols; k++) {
                    suma += at(i, k) * otra.at(k, j);
                }
                resultado.at(i, j) = suma;
            }
        }
        return resultado;
    }

    // Suma elemento a elemento: this + otra (deben tener la misma forma)
    Matrix add(const Matrix& otra) const {
        if (rows != otra.rows || cols != otra.cols)
            throw std::invalid_argument("add: las formas no coinciden");
        Matrix resultado(rows, cols);
        for (int i = 0; i < rows * cols; i++)
            resultado.data[i] = data[i] + otra.data[i];
        return resultado;
    }

    // Transpuesta: intercambia filas por columnas. [m x n] -> [n x m].
    // La vas a necesitar en attention (Q * K traspuesta).
    Matrix transpose() const {
        Matrix resultado(cols, rows);
        for (int i = 0; i < rows; i++)
            for (int j = 0; j < cols; j++)
                resultado.at(j, i) = at(i, j);
        return resultado;
    }

    // --- Utilidad: imprimir para depurar ---
    void print(const std::string& nombre = "") const {
        if (!nombre.empty()) std::cout << nombre << " ";
        std::cout << "[" << rows << " x " << cols << "]\n";
        for (int i = 0; i < rows; i++) {
            std::cout << "  ";
            for (int j = 0; j < cols; j++)
                std::cout << std::fixed << std::setprecision(3)
                          << std::setw(8) << at(i, j) << " ";
            std::cout << "\n";
        }
    }
};

#endif // MATRIX_H
