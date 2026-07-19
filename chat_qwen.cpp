// chat_qwen.cpp
// Chat interactivo con Qwen2.5. Escribes en texto normal y responde.
//
// Ya no hay que meter token IDs a mano: el encoder BPE convierte tu
// texto a tokens, y el modelo responde en streaming.

#include "qwen.h"
#include <iostream>
#include <iomanip>
#include <string>
#include <vector>
#include <random>
#include <chrono>
#include <algorithm>

int greedy(const Matrix& logits) {
    int mejor = 0;
    float mx = logits.at(0,0);
    for (int j = 1; j < logits.cols; j++)
        if (logits.at(0,j) > mx) { mx = logits.at(0,j); mejor = j; }
    return mejor;
}

int muestrear(const Matrix& logits, float temp, int top_k, std::mt19937& rng) {
    if (temp <= 0.0f) return greedy(logits);
    int V = logits.cols;
    std::vector<std::pair<float,int>> v;
    v.reserve(V);
    for (int j = 0; j < V; j++) v.push_back({logits.at(0,j)/temp, j});

    if (top_k > 0 && top_k < V) {
        std::partial_sort(v.begin(), v.begin()+top_k, v.end(),
                          [](auto&a, auto&b){ return a.first > b.first; });
        v.resize(top_k);
    }
    float mx = v[0].first;
    for (auto& p : v) mx = std::max(mx, p.first);
    float suma = 0;
    for (auto& p : v) { p.first = std::exp(p.first - mx); suma += p.first; }
    for (auto& p : v) p.first /= suma;

    std::uniform_real_distribution<float> d(0.0f, 1.0f);
    float r = d(rng), acum = 0;
    for (auto& p : v) { acum += p.first; if (r <= acum) return p.second; }
    return v.back().second;
}

int main(int argc, char** argv) {
    std::string ruta =
        "C:/Users/Said/source/repos/IA_Prueba/IA_Prueba/qwen2.5-0.5b-instruct-q4_0.gguf";
    if (argc >= 2) ruta = argv[1];

    std::cout << "=== Chat con Qwen2.5 (motor propio en C++) ===\n\n";

    Qwen modelo;
    try {
        modelo.cargar(ruta);
    } catch (const std::exception& e) {
        std::cout << "ERROR: " << e.what() << "\n";
        return 1;
    }

    if (!modelo.bpe.listo()) {
        std::cout << "\nERROR: el tokenizador BPE no se construyo.\n";
        std::cout << "El GGUF debe traer tokenizer.ggml.merges.\n";
        return 1;
    }

    // Configuracion
    float temperatura = 0.7f;
    int top_k = 40;
    int max_tokens = 200;
    std::string sistema = "You are a helpful assistant.";
    std::mt19937 rng(std::random_device{}());

    std::cout << "\nEscribe tu mensaje. Comandos: /salir  /temp <n>  /sistema <texto>\n";
    std::cout << std::string(60, '-') << "\n";

    while (true) {
        std::cout << "\n> " << std::flush;
        std::string entrada;
        if (!std::getline(std::cin, entrada)) break;
        if (entrada.empty()) continue;

        // --- Comandos ---
        if (entrada == "/salir" || entrada == "/exit") break;

        if (entrada.rfind("/temp ", 0) == 0) {
            temperatura = std::stof(entrada.substr(6));
            std::cout << "Temperatura: " << temperatura
                      << (temperatura <= 0 ? " (greedy)" : "") << "\n";
            continue;
        }

        if (entrada.rfind("/sistema ", 0) == 0) {
            sistema = entrada.substr(9);
            std::cout << "Prompt de sistema actualizado.\n";
            continue;
        }

        // --- Generar respuesta ---
        std::vector<int> prompt = modelo.armar_chat(entrada, sistema);

        auto cache = modelo.crear_cache();
        auto t1 = std::chrono::high_resolution_clock::now();

        Matrix logits = modelo.procesar_prompt(prompt, cache);

        std::cout << "\n";
        int generados = 0;
        for (int i = 0; i < max_tokens; i++) {
            int sig = muestrear(logits, temperatura, top_k, rng);

            // Fin de turno
            if (sig == modelo.eos || sig == modelo.bpe.id_de("<|im_end|>"))
                break;

            std::cout << modelo.decodificar_uno(sig) << std::flush;
            generados++;

            if (cache.tokens_procesados() >= modelo.contexto_max - 1) break;
            logits = modelo.forward(sig, cache.tokens_procesados(), cache);
        }

        auto t2 = std::chrono::high_resolution_clock::now();
        double seg = std::chrono::duration<double>(t2-t1).count();
        std::cout << "\n\n[" << prompt.size() << " tokens de prompt, "
                  << generados << " generados, "
                  << std::fixed << std::setprecision(1)
                  << (generados/seg) << " tokens/s]\n";
    }

    std::cout << "\nHasta luego.\n";
    return 0;
}
