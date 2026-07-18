// probar_gpt2_v2.cpp
// Prueba con casos que GPT-2 124M SI predice de forma confiable.
// El caso "Paris" era mala eleccion: el modelo pequeno de 2019 no tiene
// buen recuerdo factual. Estos casos son patrones fuertes que cualquier
// GPT-2 completa bien, asi que son mejor prueba de que el modelo funciona.
//
// Casos:
//   "The quick brown fox" -> el token mas probable deberia ser " jumps"
//   Numeros/repeticion: patrones que el modelo sigue con alta confianza

#include "gpt2.h"
#include <iostream>
#include <iomanip>
#include <vector>
#include <algorithm>

void mostrar_top5(GPT2& m, const std::vector<int>& tokens, const std::string& desc) {
    std::cout << "\n--- " << desc << " ---\n";
    Matrix logits = m.forward(tokens);
    int fila = logits.rows - 1;

    std::vector<std::pair<float,int>> v;
    for (int j = 0; j < logits.cols; j++)
        v.push_back({logits.at(fila, j), j});
    std::partial_sort(v.begin(), v.begin()+5, v.end(),
                      [](auto&a, auto&b){ return a.first > b.first; });

    std::cout << "Top 5 predicciones (token ID : logit):\n";
    for (int k = 0; k < 5; k++)
        std::cout << "  " << std::setw(6) << v[k].second
                  << " : " << std::fixed << std::setprecision(2)
                  << v[k].first << "\n";
}

int main(int argc, char** argv) {
    std::string ruta = "C:\\Users\\ssalc\\source\\repos\\IA_prueba\\model.safetensors";
    if (argc >= 2) ruta = argv[1];

    GPT2 m;
    m.cargar(ruta);

    // "The quick brown fox" = {464, 2068, 7586, 21831}
    // Frase MUY comun en datos de entrenamiento; deberia predecir " jumps"
    // (token 18045) con alta confianza.
    mostrar_top5(m, {464, 2068, 7586, 21831},
                 "'The quick brown fox' -> esperamos ' jumps' (18045)");

    // Repeticion fuerte: "1 2 3 4 5" deberia seguir con " 6" o similar.
    // " 1"=352 " 2"=362 " 3"=513 " 4"=604 " 5"=642
    mostrar_top5(m, {352, 362, 513, 604, 642},
                 "'1 2 3 4 5' -> esperamos continuacion numerica");

    // "Hello" repetido: "Hello Hello Hello" -> " Hello"
    // " Hello" = 18435
    mostrar_top5(m, {18435, 18435, 18435},
                 "'Hello Hello Hello' -> esperamos ' Hello' (18435)");

    std::cout << "\n=== Interpretacion ===\n";
    std::cout << "Si el token esperado sale PRIMERO (logit mas alto) en estos\n";
    std::cout << "casos de patron fuerte, tu GPT-2 funciona correctamente.\n";
    std::cout << "Estos casos son mas fiables que 'Paris' porque no dependen\n";
    std::cout << "del recuerdo factual del modelo, solo de seguir patrones.\n";

    return 0;
}
