// generar_texto.cpp
// Generacion de texto autoregresiva CON KV CACHE.
//
// Diferencia con la version original: en vez de pasar toda la secuencia
// al forward en cada paso (recalculando todo), procesamos el prompt una
// sola vez para llenar el cache, y despues cada token nuevo reutiliza
// las keys/values ya calculadas.
//
// Necesita: model.safetensors Y vocab.json de GPT-2.

#include "gpt2.h"
#include "generacion.h"
#include <iostream>
#include <iomanip>
#include <vector>
#include <random>
#include <chrono>

// Genera texto usando el KV cache.
//   prompt:      tokens iniciales
//   n_generar:   cuantos tokens nuevos producir
//   temperatura: <=0 significa greedy (determinista)
//   top_k:       0 = sin limite de candidatos
std::vector<int> generar(GPT2& modelo,
    const std::vector<int>& prompt,
    int n_generar,
    float temperatura,
    int top_k,
    std::mt19937& rng) {
    std::vector<int> seq = prompt;

    // Un cache por capa, con espacio para todo el contexto
    KVCache cache;
    cache.inicializar(modelo.num_capas, 1024, modelo.d_model);

    // Procesar el prompt completo: llena el cache y devuelve
    // los logits del ultimo token
    Matrix logits = modelo.procesar_prompt(prompt, cache);

    for (int paso = 0; paso < n_generar; paso++) {
        // Elegir el siguiente token
        int siguiente;
        if (temperatura <= 0.0f)
            siguiente = muestrear_greedy(logits);
        else
            siguiente = muestrear_temp_topk(logits, temperatura, top_k, rng);

        seq.push_back(siguiente);

        // Cortar si llegamos al limite de contexto
        if (cache.tokens_procesados() >= 1024) break;

        // Procesar SOLO el token nuevo. Su posicion es cuantos
        // tokens ya hay en el cache.
        int posicion = cache.tokens_procesados();
        logits = modelo.forward_con_cache(siguiente, posicion, cache);
    }
    return seq;
}

int main(int argc, char** argv) {
    std::string ruta_modelo = "C:/Users/Said/source/repos/IA_Prueba/IA_Prueba/model.safetensors";
    std::string ruta_vocab = "C:/Users/Said/source/repos/IA_Prueba/IA_Prueba/vocab.json";
    if (argc >= 2) ruta_modelo = argv[1];
    if (argc >= 3) ruta_vocab = argv[2];

    GPT2 modelo;
    modelo.cargar(ruta_modelo);

    Tokenizer tok;
    std::cout << "Cargando vocabulario...\n";
    tok.cargar_vocab(ruta_vocab);
    std::cout << "Vocabulario cargado (" << tok.id_a_token.size()
        << " tokens).\n\n";

    // Prompt inicial en tokens.
    // "The quick brown fox" = {464, 2068, 7586, 21831}
    // (cuando tengas el encoder BPE podras escribir texto directo)
    std::vector<int> prompt = { 464, 2068, 7586, 21831 };
    int tokens_a_generar = 50;

    float temperatura = 0.8f;
    int top_k = 40;
    std::mt19937 rng(1234);   // semilla fija = resultados reproducibles

    std::cout << "=== Generando texto (con KV cache) ===\n";
    std::cout << "Prompt: '" << tok.decodificar(prompt) << "'\n";
    std::cout << "Tokens a generar: " << tokens_a_generar << "\n\n";

    // --- Modo greedy (determinista) ---
    {
        std::cout << "Modo GREEDY (determinista):\n";
        auto inicio = std::chrono::high_resolution_clock::now();

        std::vector<int> seq = generar(modelo, prompt, tokens_a_generar,
            0.0f, 0, rng);

        auto fin = std::chrono::high_resolution_clock::now();
        double seg = std::chrono::duration<double>(fin - inicio).count();

        std::cout << "'" << tok.decodificar(seq) << "'\n";
        std::cout << "[" << std::fixed << std::setprecision(2) << seg
            << " s, " << (tokens_a_generar / seg)
            << " tokens/s]\n\n";
    }

    // --- Modo con temperatura (creativo) ---
    {
        std::cout << "Modo TEMPERATURA " << temperatura
            << " + TOP-K " << top_k << " (creativo):\n";
        auto inicio = std::chrono::high_resolution_clock::now();

        std::vector<int> seq = generar(modelo, prompt, tokens_a_generar,
            temperatura, top_k, rng);

        auto fin = std::chrono::high_resolution_clock::now();
        double seg = std::chrono::duration<double>(fin - inicio).count();

        std::cout << "'" << tok.decodificar(seq) << "'\n";
        std::cout << "[" << seg << " s, " << (tokens_a_generar / seg)
            << " tokens/s]\n\n";
    }

    std::cout << "=== Listo ===\n";
    std::cout << "Nota: greedy tiende a caer en bucles repetitivos porque\n";
    std::cout << "siempre toma el token mas probable. La temperatura rompe\n";
    std::cout << "esos ciclos al introducir aleatoriedad controlada.\n";

    return 0;
}