// paralelo.h
// FASE 1 - Optimizacion: multithreading con pool de hilos.
//
// EL PROBLEMA ORIGINAL
// El matmul usaba un solo nucleo mientras los otros 15 miraban.
//
// LA IDEA
// La multiplicacion de matrices es "vergonzosamente paralela": para
// calcular una fila (o columna) del resultado no necesitas ninguna otra.
// Son independientes, asi que cada hilo escribe en SU zona sin riesgo
// de condiciones de carrera. No hace falta ningun mutex en el calculo.
//
// POR QUE UN POOL Y NO std::thread DIRECTO
// Crear un hilo cuesta 1-3 ms en Windows. El feed-forward tarda ~8 ms,
// y se llama 48 veces por token. Crear hilos cada vez significaba pagar
// mas en administracion que en trabajo util. El pool crea los hilos una
// vez y los reutiliza.
//
// CUANTOS HILOS
// Por defecto dejamos 4 nucleos libres para que el sistema siga usable
// mientras el modelo genera. Se puede ajustar con Paralelo::configurar().

#ifndef PARALELO_H
#define PARALELO_H

#include "matrix.h"
#include "threadpool.h"
#include <thread>
#include <vector>
#include <algorithm>

namespace Paralelo {

    // Cuantos nucleos dejar libres para el sistema operativo.
    // Con 16 logicos y 4 libres, usamos 12: buen rendimiento sin
    // que la laptop se sienta trabada.
    inline int nucleos_reservados() { return 4; }

    // Numero de hilos del pool. Se calcula la primera vez y se puede
    // cambiar con configurar() antes de generar.
    inline int& num_hilos() {
        static int n = []() {
            int total = (int)std::thread::hardware_concurrency();
            if (total <= 0) total = 1;
            return std::max(1, total - nucleos_reservados());
            }();
        return n;
    }

    // El pool global. Se crea la primera vez que se usa (lazy init)
    // con el numero de hilos que haya en ese momento.
    inline ThreadPool& pool() {
        static ThreadPool p(num_hilos());
        return p;
    }

    // Cambia el numero de hilos. IMPORTANTE: llamar ANTES de la primera
    // operacion paralela, porque el pool se crea una sola vez.
    inline void configurar(int hilos) {
        num_hilos() = std::max(1, hilos);
    }

    // Debajo de este numero de operaciones no vale la pena repartir:
    // la coordinacion costaria mas que el trabajo.
    inline long long umbral_minimo() { return 20000; }
}

// ----------------------------------------------------------------------
// MATMUL PARALELO:  A * B
// ----------------------------------------------------------------------
//   A: [m x n], B: [n x p], resultado: [m x p]
//
// Reparte por FILAS si hay suficientes; si no (caso tipico con KV cache,
// donde m = 1 porque procesamos un solo token), reparte por COLUMNAS.
Matrix matmul_paralelo(const Matrix& A, const Matrix& B) {
    if (A.cols != B.rows)
        throw std::invalid_argument(
            "matmul_paralelo: columnas de A (" + std::to_string(A.cols) +
            ") deben igualar filas de B (" + std::to_string(B.rows) + ")");

    const int m = A.rows;
    const int n = A.cols;
    const int p = B.cols;

    Matrix resultado(m, p);

    const long long trabajo = (long long)m * p * n;
    const int hilos = Paralelo::num_hilos();

    // Trabajo pequeno: secuencial directo
    if (trabajo < Paralelo::umbral_minimo() || hilos <= 1) {
        for (int i = 0; i < m; i++)
            for (int j = 0; j < p; j++) {
                float suma = 0.0f;
                for (int k = 0; k < n; k++)
                    suma += A.at(i, k) * B.at(k, j);
                resultado.at(i, j) = suma;
            }
        return resultado;
    }

    // Punteros crudos: evitan el coste de at() dentro del bucle caliente
    const float* pa = A.data.data();
    const float* pb = B.data.data();
    float* pr = resultado.data.data();

    if (m >= hilos) {
        // --- Reparto por FILAS ---
        Paralelo::pool().ejecutar([&](int id) {
            int base = m / hilos, extra = m % hilos;
            int ini = id * base + std::min(id, extra);
            int fin = ini + base + (id < extra ? 1 : 0);
            for (int i = ini; i < fin; i++) {
                const float* fila_a = pa + (size_t)i * n;
                float* fila_r = pr + (size_t)i * p;
                for (int j = 0; j < p; j++) {
                    float suma = 0.0f;
                    for (int k = 0; k < n; k++)
                        suma += fila_a[k] * pb[(size_t)k * p + j];
                    fila_r[j] = suma;
                }
            }
            });
    }
    else {
        // --- Reparto por COLUMNAS (caso KV cache: una sola fila) ---
        Paralelo::pool().ejecutar([&](int id) {
            int base = p / hilos, extra = p % hilos;
            int j_ini = id * base + std::min(id, extra);
            int j_fin = j_ini + base + (id < extra ? 1 : 0);
            for (int i = 0; i < m; i++) {
                const float* fila_a = pa + (size_t)i * n;
                float* fila_r = pr + (size_t)i * p;
                for (int j = j_ini; j < j_fin; j++) {
                    float suma = 0.0f;
                    for (int k = 0; k < n; k++)
                        suma += fila_a[k] * pb[(size_t)k * p + j];
                    fila_r[j] = suma;
                }
            }
            });
    }

    return resultado;
}

// ----------------------------------------------------------------------
// MATMUL PARALELO CON TRANSPUESTA IMPLICITA:  A * B^T
// ----------------------------------------------------------------------
//   A: [m x n], B: [p x n], resultado: [m x p]
//
// POR QUE EXISTE ESTA FUNCION
// La proyeccion final hace x * wte^T, y wte es [50257 x 768]. Transponerla
// explicitamente copia 38 MILLONES de floats en cada token generado. Aqui
// leemos wte por filas sin copiar nada, y de paso el acceso a memoria es
// contiguo en ambas matrices, que es mucho mas amigable con la cache del
// procesador.
Matrix matmul_transpuesta_paralelo(const Matrix& A, const Matrix& B) {
    if (A.cols != B.cols)
        throw std::invalid_argument(
            "matmul_transpuesta: las columnas de A y B deben coincidir");

    const int m = A.rows;
    const int n = A.cols;
    const int p = B.rows;

    Matrix resultado(m, p);

    const long long trabajo = (long long)m * p * n;
    const int hilos = Paralelo::num_hilos();

    const float* pa = A.data.data();
    const float* pb = B.data.data();
    float* pr = resultado.data.data();

    if (trabajo < Paralelo::umbral_minimo() || hilos <= 1) {
        for (int i = 0; i < m; i++)
            for (int j = 0; j < p; j++) {
                const float* fa = pa + (size_t)i * n;
                const float* fb = pb + (size_t)j * n;
                float suma = 0.0f;
                for (int k = 0; k < n; k++) suma += fa[k] * fb[k];
                pr[(size_t)i * p + j] = suma;
            }
        return resultado;
    }

    if (m >= hilos) {
        Paralelo::pool().ejecutar([&](int id) {
            int base = m / hilos, extra = m % hilos;
            int ini = id * base + std::min(id, extra);
            int fin = ini + base + (id < extra ? 1 : 0);
            for (int i = ini; i < fin; i++) {
                const float* fa = pa + (size_t)i * n;
                for (int j = 0; j < p; j++) {
                    const float* fb = pb + (size_t)j * n;
                    float suma = 0.0f;
                    for (int k = 0; k < n; k++) suma += fa[k] * fb[k];
                    pr[(size_t)i * p + j] = suma;
                }
            }
            });
    }
    else {
        // Caso tipico: m = 1 (un token) y p = 50257 (el vocabulario).
        // Repartir las 50257 columnas entre los hilos.
        Paralelo::pool().ejecutar([&](int id) {
            int base = p / hilos, extra = p % hilos;
            int j_ini = id * base + std::min(id, extra);
            int j_fin = j_ini + base + (id < extra ? 1 : 0);
            for (int i = 0; i < m; i++) {
                const float* fa = pa + (size_t)i * n;
                for (int j = j_ini; j < j_fin; j++) {
                    const float* fb = pb + (size_t)j * n;
                    float suma = 0.0f;
                    for (int k = 0; k < n; k++) suma += fa[k] * fb[k];
                    pr[(size_t)i * p + j] = suma;
                }
            }
            });
    }

    return resultado;
}

#endif // PARALELO_H