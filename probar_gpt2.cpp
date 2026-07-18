// probar_gpt2.cpp
// EL MOMENTO DE LA VERDAD.
// Carga GPT-2 real, corre el forward pass completo, y verifica que
// prediga el token correcto para un caso conocido.
//
// Caso de prueba: "The capital of France is" -> deberia predecir " Paris"
//   token IDs de "The capital of France is": {464, 3139, 286, 4881, 318}
//   token esperado de " Paris": 6342
//
// Compilar: g++ -std=c++17 -O2 probar_gpt2.cpp -o probar_gpt2
// Correr:   ./probar_gpt2  C:\ruta\model.safetensors

#include "gpt2.h"
#include <iostream>
#include <iomanip>
#include <vector>
#include <algorithm>

int main(int argc, char** argv) {
    std::string ruta = "C:\\Users\\ssalc\\source\\repos\\IA_prueba\\model.safetensors";
    if (argc >= 2) ruta = argv[1];

    std::cout << "=== Prueba del forward pass de GPT-2 ===\n\n";

    GPT2 modelo;
    try {
        modelo.cargar(ruta);
    } catch (const std::exception& e) {
        std::cout << "ERROR cargando el modelo: " << e.what() << "\n";
        return 1;
    }

    // "The capital of France is"
    std::vector<int> tokens = {464, 3139, 286, 4881, 318};
    const int TOKEN_PARIS = 6342;

    std::cout << "\nEjecutando forward pass con 'The capital of France is'...\n";
    std::cout << "(sera lento: matmul sin optimizar sobre 124M de parametros)\n\n";

    Matrix logits = modelo.forward(tokens);

    std::cout << "Logits calculados. Forma: ["
              << logits.rows << " x " << logits.cols << "]\n\n";

    // El token predicho es el argmax de la ultima fila
    int predicho = modelo.argmax_ultima_fila(logits);

    std::cout << "Token predicho (ID): " << predicho << "\n";
    std::cout << "Token esperado (ID): " << TOKEN_PARIS << " (' Paris')\n\n";

    if (predicho == TOKEN_PARIS) {
        std::cout << "==================================================\n";
        std::cout << " CORRECTO! Tu GPT-2 predijo 'Paris'.\n";
        std::cout << " EL MODELO FUNCIONA DE PRINCIPIO A FIN.\n";
        std::cout << "==================================================\n";
    } else {
        std::cout << "El token no coincide. Revisemos.\n";
        // Mostrar el top-5 para diagnosticar
        int fila = logits.rows - 1;
        std::vector<std::pair<float,int>> v;
        for (int j = 0; j < logits.cols; j++)
            v.push_back({logits.at(fila, j), j});
        std::partial_sort(v.begin(), v.begin()+5, v.end(),
                          [](auto&a, auto&b){ return a.first > b.first; });
        std::cout << "\nTop 5 tokens que predijo tu modelo:\n";
        for (int k = 0; k < 5; k++)
            std::cout << "  ID " << v[k].second
                      << "  logit " << std::fixed << std::setprecision(3)
                      << v[k].first << "\n";
        std::cout << "\nSi 'Paris' (6342) aparece en el top pero no primero, "
                     "el modelo casi funciona: revisa algun detalle fino.\n";
        std::cout << "Si no aparece, hay un bug estructural: revisa la "
                     "transposicion Conv1D o el orden de carga de pesos.\n";
    }

    return 0;
}
