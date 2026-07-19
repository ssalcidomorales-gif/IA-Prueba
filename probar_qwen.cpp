// probar_qwen.cpp
// Carga Qwen2.5 desde GGUF y genera texto.
//
// NOTA SOBRE LOS TOKENS DE ENTRADA
// Todavia no tenemos encoder BPE, asi que el prompt va como IDs a mano.
// Qwen usa el formato ChatML con tokens especiales:
//   <|im_start|> = 151644
//   <|im_end|>   = 151645
//
// El formato de una conversacion es:
//   <|im_start|>system\n{sistema}<|im_end|>\n
//   <|im_start|>user\n{pregunta}<|im_end|>\n
//   <|im_start|>assistant\n

#include "qwen.h"
#include <iostream>
#include <iomanip>
#include <vector>
#include <random>
#include <chrono>
#include <algorithm>

// Sampling greedy: el token de mayor logit
int greedy(const Matrix& logits) {
    int mejor = 0;
    float mx = logits.at(0, 0);
    for (int j = 1; j < logits.cols; j++)
        if (logits.at(0, j) > mx) { mx = logits.at(0, j); mejor = j; }
    return mejor;
}

// Sampling con temperatura y top-k
int muestrear(const Matrix& logits, float temp, int top_k,
              std::mt19937& rng) {
    if (temp <= 0.0f) return greedy(logits);

    int V = logits.cols;
    std::vector<std::pair<float,int>> v;
    v.reserve(V);
    for (int j = 0; j < V; j++)
        v.push_back({logits.at(0, j) / temp, j});

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

    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    float r = dist(rng), acum = 0;
    for (auto& p : v) { acum += p.first; if (r <= acum) return p.second; }
    return v.back().second;
}

int main(int argc, char** argv) {
    std::string ruta =
        "C:/Users/Said/source/repos/IA_Prueba/IA_Prueba/qwen2.5-0.5b-instruct-q4_0.gguf";
    if (argc >= 2) ruta = argv[1];

    std::cout << "=== Qwen2.5 desde cero en C++ ===\n\n";

    Qwen modelo;
    try {
        modelo.cargar(ruta);
    } catch (const std::exception& e) {
        std::cout << "ERROR cargando el modelo: " << e.what() << "\n";
        return 1;
    }

    // Tokens del formato ChatML de Qwen
    const int IM_START = 151644;
    const int IM_END   = 151645;

    // Prompt: "<|im_start|>user\nHello<|im_end|>\n<|im_start|>assistant\n"
    // Los IDs de las palabras los sacamos del vocabulario del propio modelo.
    // "user" = 872, "\n" = 198, "Hello" = 9707, "assistant" = 77091
    std::vector<int> prompt = {
        IM_START, 872, 198,          // <|im_start|>user\n
        9707,                        // Hello
        IM_END, 198,                 // <|im_end|>\n
        IM_START, 77091, 198         // <|im_start|>assistant\n
    };

    std::cout << "\nPrompt decodificado: '"
              << modelo.decodificar(prompt) << "'\n\n";

    int a_generar = 40;
    float temperatura = 0.7f;
    int top_k = 40;
    std::mt19937 rng(1234);

    // ================== GREEDY ==================
    {
        std::cout << "--- Modo GREEDY ---\n";
        auto cache = modelo.crear_cache();
        auto t1 = std::chrono::high_resolution_clock::now();

        Matrix logits = modelo.procesar_prompt(prompt, cache);
        std::vector<int> generados;

        for (int i = 0; i < a_generar; i++) {
            int sig = greedy(logits);
            if (sig == IM_END) {           // fin de turno
                std::cout << "[<|im_end|>]";
                break;
            }
            generados.push_back(sig);
            std::cout << modelo.decodificar_uno(sig) << std::flush;
            logits = modelo.forward(sig, cache.tokens_procesados(), cache);
        }

        auto t2 = std::chrono::high_resolution_clock::now();
        double seg = std::chrono::duration<double>(t2-t1).count();
        std::cout << "\n[" << std::fixed << std::setprecision(2) << seg
                  << " s, " << (generados.size()/seg) << " tokens/s]\n\n";
    }

    // ================== TEMPERATURA ==================
    {
        std::cout << "--- Modo TEMPERATURA " << temperatura
                  << " + TOP-K " << top_k << " ---\n";
        auto cache = modelo.crear_cache();
        auto t1 = std::chrono::high_resolution_clock::now();

        Matrix logits = modelo.procesar_prompt(prompt, cache);
        std::vector<int> generados;

        for (int i = 0; i < a_generar; i++) {
            int sig = muestrear(logits, temperatura, top_k, rng);
            if (sig == IM_END) {
                std::cout << "[<|im_end|>]";
                break;
            }
            generados.push_back(sig);
            std::cout << modelo.decodificar_uno(sig) << std::flush;
            logits = modelo.forward(sig, cache.tokens_procesados(), cache);
        }

        auto t2 = std::chrono::high_resolution_clock::now();
        double seg = std::chrono::duration<double>(t2-t1).count();
        std::cout << "\n[" << seg << " s, "
                  << (generados.size()/seg) << " tokens/s]\n\n";
    }

    std::cout << "=== Listo ===\n";
    std::cout << "Si el texto tiene sentido, Qwen2.5 funciona: RMSNorm,\n";
    std::cout << "RoPE, GQA y SwiGLU implementados desde cero, cargando\n";
    std::cout << "pesos cuantizados en Q4_0.\n";

    return 0;
}
