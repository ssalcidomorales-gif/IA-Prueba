// generar_texto.cpp
// Generacion de texto autoregresiva: dale tokens iniciales y GPT-2
// va prediciendo y agregando tokens uno por uno, formando texto.
//
// Necesita: model.safetensors Y vocab.json (ambos de GPT-2).
// vocab.json se descarga de:
//   https://huggingface.co/openai-community/gpt2/resolve/main/vocab.json
//
// Compilar con todos los headers. Correr:
//   ./generar_texto  ruta\model.safetensors  ruta\vocab.json

#include "gpt2.h"
#include "generacion.h"
#include <iostream>
#include <vector>
#include <random>

int main(int argc, char** argv) {
    std::string ruta_modelo = "C:\\Users\\ssalc\\source\\repos\\IA_prueba\\model.safetensors";
    std::string ruta_vocab  = "C:\\Users\\ssalc\\source\\repos\\IA_prueba\\vocab.json";
    if (argc >= 2) ruta_modelo = argv[1];
    if (argc >= 3) ruta_vocab = argv[2];

    GPT2 modelo;
    modelo.cargar(ruta_modelo);

    Tokenizer tok;
    std::cout << "Cargando vocabulario...\n";
    tok.cargar_vocab(ruta_vocab);
    std::cout << "Vocabulario cargado (" << tok.id_a_token.size() << " tokens).\n\n";

    // Prompt inicial en tokens. "The quick brown fox" = {464, 2068, 7586, 21831}
    // (mas adelante, cuando tengas el encoder BPE, escribiras texto directo)
    std::vector<int> tokens = {464, 2068, 7586, 21831};
    int tokens_a_generar = 30;

    // Config de sampling
    float temperatura = 0.8f;
    int top_k = 40;
    std::mt19937 rng(1234);  // semilla fija para reproducibilidad

    std::cout << "=== Generando texto ===\n";
    std::cout << "Prompt inicial: '" << tok.decodificar(tokens) << "'\n\n";
    std::cout << "Modo GREEDY (determinista):\n";

    // --- Generacion greedy ---
    {
        std::vector<int> seq = tokens;
        for (int paso = 0; paso < tokens_a_generar; paso++) {
            Matrix logits = modelo.forward(seq);
            int siguiente = muestrear_greedy(logits);
            seq.push_back(siguiente);
        }
        std::cout << "'" << tok.decodificar(seq) << "'\n\n";
    }

    std::cout << "Modo TEMPERATURA " << temperatura
              << " + TOP-K " << top_k << " (creativo):\n";

    // --- Generacion con temperatura ---
    {
        std::vector<int> seq = tokens;
        for (int paso = 0; paso < tokens_a_generar; paso++) {
            Matrix logits = modelo.forward(seq);
            int siguiente = muestrear_temp_topk(logits, temperatura, top_k, rng);
            seq.push_back(siguiente);
        }
        std::cout << "'" << tok.decodificar(seq) << "'\n\n";
    }

    std::cout << "=== Listo ===\n";
    std::cout << "Nota: sera lento (cada token nuevo = un forward pass completo\n";
    std::cout << "sin optimizar). La velocidad es el tema de la Fase 1.\n";

    return 0;
}
