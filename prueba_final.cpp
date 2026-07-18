// prueba_final.cpp
// Verifica que el modelo funciona midiendo la CONFIANZA (diferencia entre
// el logit ganador y el promedio), no el valor absoluto de los logits.
// En un softmax solo importan las DIFERENCIAS entre logits, no su valor
// absoluto: sumar una constante a todos no cambia las probabilidades.

#include "gpt2.h"
#include <iostream>
#include <iomanip>
#include <vector>
#include <algorithm>
#include <cmath>

void analizar(GPT2& m, const std::vector<int>& tokens,
              int token_esperado, const std::string& desc) {
    Matrix logits = m.forward(tokens);
    int fila = logits.rows - 1;

    // Estadisticas de la fila de logits
    float maxv = -1e30f, suma = 0;
    int argmax = 0;
    for (int j = 0; j < logits.cols; j++) {
        float v = logits.at(fila, j);
        suma += v;
        if (v > maxv) { maxv = v; argmax = j; }
    }
    float media = suma / logits.cols;
    float logit_esperado = logits.at(fila, token_esperado);

    std::cout << "\n--- " << desc << " ---\n";
    std::cout << "  Token ganador: " << argmax
              << "  (esperado: " << token_esperado << ")\n";
    std::cout << "  Logit ganador:   " << std::fixed << std::setprecision(2) << maxv << "\n";
    std::cout << "  Logit esperado:  " << logit_esperado << "\n";
    std::cout << "  Media de logits: " << media << "\n";
    std::cout << "  Confianza del esperado (logit - media): "
              << (logit_esperado - media) << "\n";

    if (argmax == token_esperado)
        std::cout << "  --> CORRECTO: el token esperado gana.\n";
    else {
        // que tan cerca esta del ganador?
        std::cout << "  --> El esperado no gana, diferencia con el ganador: "
                  << (maxv - logit_esperado) << " logits\n";
    }
}

int main(int argc, char** argv) {
    std::string ruta = "C:\\Users\\ssalc\\source\\repos\\IA_prueba\\model.safetensors";
    if (argc >= 2) ruta = argv[1];

    GPT2 m;
    m.cargar(ruta);

    std::cout << "\n=== Prueba de confianza del modelo ===\n";
    std::cout << "Lo que importa es la CONFIANZA (logit - media), no el valor\n";
    std::cout << "absoluto. Un valor alto y positivo = el modelo esta seguro.\n";

    // Caso 1: secuencia numerica, " 6" = token 718. Patron fortisimo.
    analizar(m, {352, 362, 513, 604, 642}, 718,
             "'1 2 3 4 5' -> ' 6' (718)");

    // Caso 2: "The quick brown fox" -> " jumps" (18045)
    // GPT-2 124M lo predice, aunque compite con otras continuaciones.
    analizar(m, {464, 2068, 7586, 21831}, 18045,
             "'The quick brown fox' -> ' jumps' (18045)");

    // Caso 3: alfabeto "a b c d" -> " e" = 304
    // " a"=257 " b"=275 " c"=269 " d"=288 " e"=304
    analizar(m, {257, 275, 269, 288}, 304,
             "'a b c d' -> ' e' (304)");

    std::cout << "\n=== Conclusion ===\n";
    std::cout << "Si en los casos de patron fuerte (numeros, alfabeto) el token\n";
    std::cout << "esperado GANA con confianza positiva alta, el modelo funciona.\n";
    std::cout << "Los logits negativos en su valor absoluto NO son un problema:\n";
    std::cout << "el softmax solo mira diferencias, no valores absolutos.\n";

    return 0;
}
