// diagnostico_capas.cpp
// Instrumenta el forward pass para ver la magnitud de los valores en cada
// etapa. Si la senal se ve sana hasta cierto punto y luego se aplasta,
// ahi esta el bug. Es el "logit lens" de depuracion.
//
// Compilar con los mismos headers. Correr con la ruta del modelo.

#include "gpt2.h"
#include <iostream>
#include <iomanip>
#include <cmath>

// Devuelve la magnitud promedio (valor absoluto medio) de una matriz
float magnitud(const Matrix& m) {
    float suma = 0;
    for (float v : m.data) suma += std::fabs(v);
    return suma / m.data.size();
}

// Devuelve el valor maximo absoluto
float maxabs(const Matrix& m) {
    float mx = 0;
    for (float v : m.data) mx = std::max(mx, std::fabs(v));
    return mx;
}

int main(int argc, char** argv) {
    std::string ruta = "C:\\Users\\ssalc\\source\\repos\\IA_prueba\\model.safetensors";
    if (argc >= 2) ruta = argv[1];

    GPT2 m;
    m.cargar(ruta);

    std::vector<int> tokens = {464, 3139, 286, 4881, 318};

    std::cout << "\n=== Magnitud de la senal en cada etapa ===\n";
    std::cout << "(magnitud = promedio de valores absolutos)\n\n";

    // Replicar el forward paso a paso, imprimiendo magnitudes
    Matrix x = embeddings(tokens, m.wte, m.wpe);
    std::cout << std::left << std::setw(30) << "tras embeddings"
              << "mag=" << std::fixed << std::setprecision(3) << magnitud(x)
              << "  max=" << maxabs(x) << "\n";

    for (int i = 0; i < m.num_capas; i++) {
        x = bloque_transformer(x, m.bloques[i], m.num_cabezas);
        std::cout << std::left << std::setw(30)
                  << ("tras bloque " + std::to_string(i))
                  << "mag=" << magnitud(x) << "  max=" << maxabs(x) << "\n";
    }

    Matrix xn = layernorm(x, m.ln_f_w, m.ln_f_b);
    std::cout << std::left << std::setw(30) << "tras layernorm final"
              << "mag=" << magnitud(xn) << "  max=" << maxabs(xn) << "\n";

    Matrix logits = xn.matmul(m.wte.transpose());
    std::cout << std::left << std::setw(30) << "logits finales"
              << "mag=" << magnitud(logits) << "  max=" << maxabs(logits) << "\n";

    std::cout << "\nReferencia sana: embeddings mag ~0.1, deberia CRECER "
                 "gradualmente por capa hasta mag ~5-20 antes del ln final.\n";
    std::cout << "Si se mantiene plana o colapsa, ahi esta el problema.\n";

    return 0;
}
