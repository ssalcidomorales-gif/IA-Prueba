// test_layers.cpp
// Verifica embeddings y layernorm con casos calculados a mano.
// Compilar: g++ -std=c++17 -O2 test_layers.cpp -o test_layers
// Correr:   ./test_layers

#include "layers.h"
#include <iostream>
#include <cmath>

int pruebas_ok = 0, pruebas_totales = 0;

bool casi_igual(float a, float b, float tol = 1e-4f) {
    return std::fabs(a - b) < tol;
}

void verificar(const std::string& nombre, const Matrix& obtenida, const Matrix& esperada) {
    pruebas_totales++;
    if (obtenida.rows != esperada.rows || obtenida.cols != esperada.cols) {
        std::cout << "[FALLO] " << nombre << ": forma incorrecta\n";
        return;
    }
    for (int i = 0; i < obtenida.rows; i++)
        for (int j = 0; j < obtenida.cols; j++)
            if (!casi_igual(obtenida.at(i, j), esperada.at(i, j))) {
                std::cout << "[FALLO] " << nombre << ": en (" << i << "," << j
                          << ") obtuve " << obtenida.at(i, j)
                          << " esperaba " << esperada.at(i, j) << "\n";
                return;
            }
    std::cout << "[OK]    " << nombre << "\n";
    pruebas_ok++;
}

void afirmar(const std::string& nombre, bool cond) {
    pruebas_totales++;
    if (cond) { std::cout << "[OK]    " << nombre << "\n"; pruebas_ok++; }
    else      { std::cout << "[FALLO] " << nombre << "\n"; }
}

int main() {
    std::cout << "=== Tests de embeddings y layernorm ===\n\n";

    // ---------------- EMBEDDINGS ----------------
    // wte (vocabulario de 3 tokens, dimension 4):
    //   token 0: [1, 1, 1, 1]
    //   token 1: [2, 2, 2, 2]
    //   token 2: [3, 3, 3, 3]
    // wpe (posiciones, dimension 4):
    //   pos 0: [10, 20, 30, 40]
    //   pos 1: [50, 60, 70, 80]
    {
        Matrix wte = {{1,1,1,1}, {2,2,2,2}, {3,3,3,3}};
        Matrix wpe = {{10,20,30,40}, {50,60,70,80}};

        // Tokens {2, 0}:
        //   pos 0 -> wte[2] + wpe[0] = [3,3,3,3] + [10,20,30,40] = [13,23,33,43]
        //   pos 1 -> wte[0] + wpe[1] = [1,1,1,1] + [50,60,70,80] = [51,61,71,81]
        std::vector<int> tokens = {2, 0};
        Matrix esperada = {{13,23,33,43}, {51,61,71,81}};
        verificar("embeddings {2,0}", embeddings(tokens, wte, wpe), esperada);
    }

    // La forma de salida debe ser [num_tokens x d]
    {
        Matrix wte = {{1,1,1,1}, {2,2,2,2}, {3,3,3,3}};
        Matrix wpe = {{10,20,30,40}, {50,60,70,80}};
        std::vector<int> tokens = {1};
        Matrix r = embeddings(tokens, wte, wpe);
        afirmar("embeddings forma [1 x 4]", r.rows == 1 && r.cols == 4);
    }

    // ---------------- LAYERNORM ----------------
    // Fila [1, 2, 3, 4]:
    //   media = 2.5
    //   var   = ((1-2.5)^2+(2-2.5)^2+(3-2.5)^2+(4-2.5)^2)/4
    //         = (2.25+0.25+0.25+2.25)/4 = 5/4 = 1.25
    //   std   = sqrt(1.25 + 1e-5) ~= 1.11803
    //   norm  = [(1-2.5), (2-2.5), (3-2.5), (4-2.5)] / 1.11803
    //         = [-1.3416, -0.4472, 0.4472, 1.3416]
    // Con gamma=1, beta=0 la salida es igual a norm.
    {
        Matrix x     = {{1, 2, 3, 4}};
        Matrix gamma = {{1, 1, 1, 1}};
        Matrix beta  = {{0, 0, 0, 0}};
        Matrix esperada = {{-1.34164f, -0.44721f, 0.44721f, 1.34164f}};
        verificar("layernorm basico", layernorm(x, gamma, beta), esperada);
    }

    // Con gamma y beta distintos: salida = norm * gamma + beta
    // norm = [-1.34164, -0.44721, 0.44721, 1.34164]
    // gamma = [2,2,2,2], beta = [1,1,1,1]
    // salida = norm*2 + 1 = [-1.68328, 0.10557, 1.89443, 3.68328]
    {
        Matrix x     = {{1, 2, 3, 4}};
        Matrix gamma = {{2, 2, 2, 2}};
        Matrix beta  = {{1, 1, 1, 1}};
        Matrix esperada = {{-1.68328f, 0.10557f, 1.89443f, 3.68328f}};
        verificar("layernorm con gamma/beta", layernorm(x, gamma, beta), esperada);
    }

    // Propiedad: tras normalizar (gamma=1,beta=0), cada fila tiene media ~0
    {
        Matrix x     = {{5, 10, 15, 20}, {-3, -1, 1, 3}};
        Matrix gamma = {{1, 1, 1, 1}};
        Matrix beta  = {{0, 0, 0, 0}};
        Matrix r = layernorm(x, gamma, beta);
        bool medias_cero = true;
        for (int i = 0; i < r.rows; i++) {
            float m = 0;
            for (int j = 0; j < r.cols; j++) m += r.at(i, j);
            m /= r.cols;
            if (std::fabs(m) > 1e-4f) medias_cero = false;
        }
        afirmar("layernorm deja media ~0 por fila", medias_cero);
    }

    std::cout << "\n=== " << pruebas_ok << "/" << pruebas_totales
              << " pruebas pasaron ===\n";
    if (pruebas_ok == pruebas_totales)
        std::cout << "Embeddings y layernorm solidos. Listo para attention.\n";
    else
        std::cout << "Hay fallos, revisa antes de seguir.\n";

    return (pruebas_ok == pruebas_totales) ? 0 : 1;
}
