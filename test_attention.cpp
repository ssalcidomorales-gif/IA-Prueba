// test_attention.cpp
// Verifica una cabeza de atencion. El foco esta en la mascara causal:
// comprobar que un token NO recibe informacion de tokens futuros.
// Compilar: g++ -std=c++17 -O2 test_attention.cpp -o test_attention

#include "attention.h"
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
    std::cout << "=== Tests de una cabeza de atencion ===\n\n";

    // --- Test 1: la mascara causal pone -inf donde j > i ---
    {
        Matrix scores = {{1, 2, 3},
                         {4, 5, 6},
                         {7, 8, 9}};
        aplicar_mascara_causal(scores);
        // Esperado: triangular inferior intacto, superior en -1e10
        bool ok =
            scores.at(0,0)==1 && scores.at(0,1)<-1e9 && scores.at(0,2)<-1e9 &&
            scores.at(1,0)==4 && scores.at(1,1)==5   && scores.at(1,2)<-1e9 &&
            scores.at(2,0)==7 && scores.at(2,1)==8   && scores.at(2,2)==9;
        afirmar("mascara causal enmascara el futuro", ok);
    }

    // --- Test 2: el PRIMER token solo se atiende a si mismo ---
    // Con la mascara, el token 0 no ve a nadie mas. Por tanto su salida
    // debe ser EXACTAMENTE el value del token 0, sin importar los demas.
    {
        // dim_cabeza = 2, tres tokens
        Matrix Q = {{1, 0}, {0, 1}, {1, 1}};
        Matrix K = {{1, 0}, {0, 1}, {1, 1}};
        // Values bien distintos para notar si se "cuela" informacion
        Matrix V = {{10, 20},      // value del token 0
                    {30, 40},      // value del token 1 (futuro para token 0)
                    {50, 60}};     // value del token 2 (futuro para token 0)

        Matrix salida = atencion_una_cabeza(Q, K, V);

        // La fila 0 de la salida debe ser igual al value del token 0: [10, 20]
        afirmar("token 0 solo recibe su propio value",
                casi_igual(salida.at(0,0), 10.0f) &&
                casi_igual(salida.at(0,1), 20.0f));
    }

    // --- Test 3: la forma de salida es [n x dim_cabeza] ---
    {
        Matrix Q(4, 2), K(4, 2), V(4, 2);
        Matrix salida = atencion_una_cabeza(Q, K, V);
        afirmar("salida tiene forma [4 x 2]",
                salida.rows == 4 && salida.cols == 2);
    }

    // --- Test 4: cada salida es una MEZCLA de values pasados ---
    // Si todos los Q y K son iguales, los scores (antes de mascara) son
    // iguales, asi que cada token reparte atencion uniforme entre los que
    // puede ver. El ultimo token ve a los 3, con pesos 1/3 cada uno.
    // Su salida = (V0 + V1 + V2) / 3
    {
        Matrix Q = {{1, 1}, {1, 1}, {1, 1}};
        Matrix K = {{1, 1}, {1, 1}, {1, 1}};
        Matrix V = {{3, 0}, {0, 3}, {3, 3}};
        Matrix salida = atencion_una_cabeza(Q, K, V);
        // ultimo token: promedio de los 3 values = [(3+0+3)/3, (0+3+3)/3] = [2, 2]
        afirmar("ultimo token promedia los 3 values",
                casi_igual(salida.at(2,0), 2.0f) &&
                casi_igual(salida.at(2,1), 2.0f));
        // primer token: solo se ve a si mismo = [3, 0]
        afirmar("con atencion uniforme, token 0 sigue aislado",
                casi_igual(salida.at(0,0), 3.0f) &&
                casi_igual(salida.at(0,1), 0.0f));
    }

    std::cout << "\n=== " << pruebas_ok << "/" << pruebas_totales
              << " pruebas pasaron ===\n";
    if (pruebas_ok == pruebas_totales)
        std::cout << "Una cabeza de atencion funciona. La causalidad se respeta.\n";

    return (pruebas_ok == pruebas_totales) ? 0 : 1;
}
