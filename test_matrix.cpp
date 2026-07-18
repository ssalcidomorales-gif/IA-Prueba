// test_matrix.cpp
// Verifica cada operacion de Matrix contra resultados calculados A MANO.
// Este habito -probar cada ladrillo aislado- es lo que te salva en attention.
// Compilar:  g++ -std=c++17 -O2 test_matrix.cpp -o test_matrix
// Correr:    ./test_matrix

#include "matrix.h"
#include <iostream>

int pruebas_totales = 0;
int pruebas_ok = 0;

// Compara dos floats permitiendo un error minusculo (los float no son exactos)
bool casi_igual(float a, float b, float tol = 1e-4f) {
    return std::fabs(a - b) < tol;
}

// Verifica que una matriz coincida con los valores esperados
void verificar(const std::string& nombre, const Matrix& obtenida, const Matrix& esperada) {
    pruebas_totales++;
    if (obtenida.rows != esperada.rows || obtenida.cols != esperada.cols) {
        std::cout << "[FALLO] " << nombre << ": forma incorrecta. Obtenida ["
                  << obtenida.rows << "x" << obtenida.cols << "], esperada ["
                  << esperada.rows << "x" << esperada.cols << "]\n";
        return;
    }
    for (int i = 0; i < obtenida.rows; i++) {
        for (int j = 0; j < obtenida.cols; j++) {
            if (!casi_igual(obtenida.at(i, j), esperada.at(i, j))) {
                std::cout << "[FALLO] " << nombre << ": en (" << i << "," << j
                          << ") obtuve " << obtenida.at(i, j)
                          << " pero esperaba " << esperada.at(i, j) << "\n";
                return;
            }
        }
    }
    std::cout << "[OK]    " << nombre << "\n";
    pruebas_ok++;
}

int main() {
    std::cout << "=== Tests de la clase Matrix ===\n\n";

    // --- Test 1: matmul 2x2 (calculado a mano) ---
    // A = [1 2]   B = [5 6]
    //     [3 4]       [7 8]
    //
    // A*B = [1*5+2*7  1*6+2*8]  =  [19 22]
    //       [3*5+4*7  3*6+4*8]     [43 50]
    {
        Matrix A = {{1, 2}, {3, 4}};
        Matrix B = {{5, 6}, {7, 8}};
        Matrix esperada = {{19, 22}, {43, 50}};
        verificar("matmul 2x2", A.matmul(B), esperada);
    }

    // --- Test 2: matmul rectangular [2x3] * [3x2] = [2x1... no, 2x2] ---
    // A = [1 2 3]   B = [7  8 ]
    //     [4 5 6]       [9  10]
    //                   [11 12]
    // A*B = [1*7+2*9+3*11   1*8+2*10+3*12]  =  [58  64 ]
    //       [4*7+5*9+6*11   4*8+5*10+6*12]     [139 154]
    {
        Matrix A = {{1, 2, 3}, {4, 5, 6}};
        Matrix B = {{7, 8}, {9, 10}, {11, 12}};
        Matrix esperada = {{58, 64}, {139, 154}};
        verificar("matmul [2x3]*[3x2]", A.matmul(B), esperada);
    }

    // --- Test 3: identidad. A * I = A ---
    {
        Matrix A = {{2, -1}, {0, 3}};
        Matrix I = {{1, 0}, {0, 1}};
        verificar("A * identidad = A", A.matmul(I), A);
    }

    // --- Test 4: suma elemento a elemento ---
    // [1 2] + [10 20] = [11 22]
    // [3 4]   [30 40]   [33 44]
    {
        Matrix A = {{1, 2}, {3, 4}};
        Matrix B = {{10, 20}, {30, 40}};
        Matrix esperada = {{11, 22}, {33, 44}};
        verificar("add 2x2", A.add(B), esperada);
    }

    // --- Test 5: transpuesta [2x3] -> [3x2] ---
    // A = [1 2 3]      A^T = [1 4]
    //     [4 5 6]            [2 5]
    //                        [3 6]
    {
        Matrix A = {{1, 2, 3}, {4, 5, 6}};
        Matrix esperada = {{1, 4}, {2, 5}, {3, 6}};
        verificar("transpose [2x3]", A.transpose(), esperada);
    }

    // --- Test 6: transponer dos veces devuelve el original ---
    {
        Matrix A = {{1, 2, 3}, {4, 5, 6}};
        verificar("transpose doble = original", A.transpose().transpose(), A);
    }

    // --- Test 7: propiedad (A*B)^T == B^T * A^T ---
    // Un chequeo matematico real: si esto pasa, matmul y transpose
    // estan bien coordinados. Es la clase de verificacion que usaras
    // constantemente contra la referencia de Python.
    {
        Matrix A = {{1, 2, 3}, {4, 5, 6}};
        Matrix B = {{7, 8}, {9, 10}, {11, 12}};
        Matrix lado_izq = A.matmul(B).transpose();
        Matrix lado_der = B.transpose().matmul(A.transpose());
        verificar("(A*B)^T == B^T * A^T", lado_izq, lado_der);
    }

    // --- Resumen ---
    std::cout << "\n=== " << pruebas_ok << "/" << pruebas_totales
              << " pruebas pasaron ===\n";

    if (pruebas_ok == pruebas_totales) {
        std::cout << "Todo bien. Los ladrillos son solidos, "
                     "puedes construir encima.\n";
        return 0;
    } else {
        std::cout << "Hay fallos. NO sigas hasta arreglarlos: "
                     "todo lo demas depende de esto.\n";
        return 1;
    }
}
