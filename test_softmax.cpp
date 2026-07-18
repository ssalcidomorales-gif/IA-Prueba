// test_softmax.cpp
// Verifica softmax: valores conocidos, que cada fila sume 1, y sobre todo
// que NO explote con numeros grandes (la prueba de estabilidad).
// Compilar: g++ -std=c++17 -O2 test_softmax.cpp -o test_softmax

#include "softmax.h"
#include <iostream>
#include <cmath>

int pruebas_ok = 0, pruebas_totales = 0;

bool casi_igual(float a, float b, float tol = 1e-4f) {
    return std::fabs(a - b) < tol;
}

void afirmar(const std::string& nombre, bool cond) {
    pruebas_totales++;
    if (cond) { std::cout << "[OK]    " << nombre << "\n"; pruebas_ok++; }
    else      { std::cout << "[FALLO] " << nombre << "\n"; }
}

int main() {
    std::cout << "=== Tests de softmax ===\n\n";

    // Test 1: softmax de [0, 0, 0] -> [1/3, 1/3, 1/3]
    // Valores iguales => probabilidades iguales.
    {
        Matrix x = {{0, 0, 0}};
        Matrix r = softmax(x);
        afirmar("softmax [0,0,0] = [1/3,1/3,1/3]",
                casi_igual(r.at(0,0), 1.0f/3) &&
                casi_igual(r.at(0,1), 1.0f/3) &&
                casi_igual(r.at(0,2), 1.0f/3));
    }

    // Test 2: cada fila suma 1
    {
        Matrix x = {{1, 2, 3, 4}, {-5, 0, 5, 10}};
        Matrix r = softmax(x);
        bool suman_uno = true;
        for (int i = 0; i < r.rows; i++) {
            float s = 0;
            for (int j = 0; j < r.cols; j++) s += r.at(i, j);
            if (!casi_igual(s, 1.0f)) suman_uno = false;
        }
        afirmar("cada fila suma 1", suman_uno);
    }

    // Test 3: el valor mas grande recibe la probabilidad mas alta
    {
        Matrix x = {{1, 5, 2}};  // el indice 1 es el mayor
        Matrix r = softmax(x);
        afirmar("el mayor tiene mas probabilidad",
                r.at(0,1) > r.at(0,0) && r.at(0,1) > r.at(0,2));
    }

    // Test 4: valores conocidos de softmax [1, 2, 3]
    // exp(1)=2.71828, exp(2)=7.38906, exp(3)=20.0855
    // suma = 30.1928
    // -> [0.09003, 0.24473, 0.66524]
    {
        Matrix x = {{1, 2, 3}};
        Matrix r = softmax(x);
        afirmar("softmax [1,2,3] valores correctos",
                casi_igual(r.at(0,0), 0.09003f) &&
                casi_igual(r.at(0,1), 0.24473f) &&
                casi_igual(r.at(0,2), 0.66524f));
    }

    // Test 5: ESTABILIDAD. Con numeros enormes, sin el truco del max
    // esto daria exp(1000)=infinito y luego NaN. Con el truco, funciona.
    {
        Matrix x = {{1000, 1001, 1002}};
        Matrix r = softmax(x);
        // La diferencia entre valores es la misma que [0,1,2], asi que
        // el resultado debe ser identico a softmax([0,1,2]).
        // exp(0)=1, exp(1)=2.71828, exp(2)=7.38906, suma=11.1073
        // -> [0.09003, 0.24473, 0.66524]
        bool sin_nan = !std::isnan(r.at(0,0)) &&
                       !std::isnan(r.at(0,1)) &&
                       !std::isnan(r.at(0,2));
        bool correcto = casi_igual(r.at(0,0), 0.09003f) &&
                        casi_igual(r.at(0,1), 0.24473f) &&
                        casi_igual(r.at(0,2), 0.66524f);
        afirmar("estabilidad: numeros grandes sin NaN", sin_nan && correcto);
    }

    std::cout << "\n=== " << pruebas_ok << "/" << pruebas_totales
              << " pruebas pasaron ===\n";
    if (pruebas_ok == pruebas_totales)
        std::cout << "Softmax solido y estable. Siguiente: una cabeza de atencion.\n";

    return (pruebas_ok == pruebas_totales) ? 0 : 1;
}
