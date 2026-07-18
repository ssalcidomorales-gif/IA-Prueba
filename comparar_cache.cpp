// comparar_cache.cpp
// Verifica que la version con KV cache produce EXACTAMENTE el mismo texto
// que la version original, y mide cuanto mas rapida es.
//
// La prueba es directa: mismo prompt, mismo metodo de sampling (greedy,
// que es determinista), y comparamos token por token. Si un solo token
// difiere, hay un bug en el cache.

#include "gpt2.h"
#include "generacion.h"
#include <iostream>
#include <iomanip>
#include <vector>
#include <chrono>

int main(int argc, char** argv) {
    std::string ruta_modelo = "C:/Users/Said/source/repos/IA_Prueba/IA_Prueba/model.safetensors";
    std::string ruta_vocab  = "C:/Users/Said/source/repos/IA_Prueba/IA_Prueba/vocab.json";
    if (argc >= 2) ruta_modelo = argv[1];
    if (argc >= 3) ruta_vocab  = argv[2];

    GPT2 modelo;
    modelo.cargar(ruta_modelo);

    Tokenizer tok;
    std::cout << "Cargando vocabulario...\n";
    tok.cargar_vocab(ruta_vocab);
    std::cout << "Vocabulario cargado.\n\n";

    // "The quick brown fox"
    std::vector<int> prompt = {464, 2068, 7586, 21831};
    const int A_GENERAR = 100;

    std::cout << "=== Comparacion: original vs KV cache ===\n";
    std::cout << "Prompt: '" << tok.decodificar(prompt) << "'\n";
    std::cout << "Tokens a generar: " << A_GENERAR << "\n\n";

    // ==================================================================
    // METODO 1: original (recalcula todo en cada paso)
    // ==================================================================
    std::cout << "Metodo ORIGINAL (sin cache)...\n";
    std::vector<int> seq_original = prompt;

    auto inicio1 = std::chrono::high_resolution_clock::now();
    for (int paso = 0; paso < A_GENERAR; paso++) {
        Matrix logits = modelo.forward(seq_original);
        int siguiente = muestrear_greedy(logits);
        seq_original.push_back(siguiente);
    }
    auto fin1 = std::chrono::high_resolution_clock::now();
    double seg_original =
        std::chrono::duration<double>(fin1 - inicio1).count();

    std::cout << "  Tiempo: " << std::fixed << std::setprecision(2)
              << seg_original << " s\n\n";

    // ==================================================================
    // METODO 2: con KV cache (procesa un token a la vez)
    // ==================================================================
    std::cout << "Metodo CON CACHE...\n";
    std::vector<int> seq_cache = prompt;

    KVCache cache;
    cache.inicializar(modelo.num_capas, 1024, modelo.d_model);

    auto inicio2 = std::chrono::high_resolution_clock::now();

    // Llenar el cache con el prompt inicial
    Matrix logits = modelo.procesar_prompt(prompt, cache);

    // Generar: cada paso procesa UN solo token
    for (int paso = 0; paso < A_GENERAR; paso++) {
        int siguiente = muestrear_greedy(logits);
        seq_cache.push_back(siguiente);

        // La posicion del token nuevo es cuantos ya procesamos
        int posicion = cache.tokens_procesados();
        logits = modelo.forward_con_cache(siguiente, posicion, cache);
    }
    auto fin2 = std::chrono::high_resolution_clock::now();
    double seg_cache =
        std::chrono::duration<double>(fin2 - inicio2).count();

    std::cout << "  Tiempo: " << seg_cache << " s\n\n";

    // ==================================================================
    // VERIFICACION: los textos deben ser IDENTICOS
    // ==================================================================
    std::cout << "=== Resultados ===\n\n";
    std::cout << "Original: '" << tok.decodificar(seq_original) << "'\n\n";
    std::cout << "Cache:    '" << tok.decodificar(seq_cache) << "'\n\n";

    bool identicos = (seq_original.size() == seq_cache.size());
    int primer_fallo = -1;
    if (identicos) {
        for (size_t i = 0; i < seq_original.size(); i++) {
            if (seq_original[i] != seq_cache[i]) {
                identicos = false;
                primer_fallo = (int)i;
                break;
            }
        }
    }

    std::cout << "=== Verificacion ===\n";
    if (identicos) {
        std::cout << "CORRECTO: los " << seq_original.size()
                  << " tokens coinciden exactamente.\n";
        std::cout << "El KV cache no cambio el resultado.\n\n";

        std::cout << "=== Rendimiento ===\n";
        std::cout << "  Sin cache: " << seg_original << " s\n";
        std::cout << "  Con cache: " << seg_cache << " s\n";
        if (seg_cache > 0) {
            std::cout << "  Mejora:    " << std::setprecision(2)
                      << (seg_original / seg_cache) << "x mas rapido\n";
        }
        std::cout << "\nLa ventaja crece con la longitud: sin cache el costo\n";
        std::cout << "es O(n^2), con cache es O(n).\n";
    } else {
        std::cout << "FALLO: los tokens difieren.\n";
        if (primer_fallo >= 0) {
            std::cout << "Primera diferencia en la posicion " << primer_fallo
                      << ":\n";
            std::cout << "  original: token " << seq_original[primer_fallo]
                      << " ('" << tok.decodificar_uno(seq_original[primer_fallo])
                      << "')\n";
            std::cout << "  cache:    token " << seq_cache[primer_fallo]
                      << " ('" << tok.decodificar_uno(seq_cache[primer_fallo])
                      << "')\n";
        }
        std::cout << "\nSospechosos habituales:\n";
        std::cout << "  - la posicion del embedding posicional se desalineo\n";
        std::cout << "  - el cache no acumula bien las K/V\n";
        std::cout << "  - se estan leyendo mas filas del cache de las validas\n";
    }

    return identicos ? 0 : 1;
}
