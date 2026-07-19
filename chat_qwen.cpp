// chat_qwen.cpp
// Chat interactivo con Qwen2.5, con penalizacion por repeticion
// y salida UTF-8 correcta en Windows.

#include "qwen.h"
#include "sampling.h"
#include <iostream>
#include <iomanip>
#include <string>
#include <vector>
#include <chrono>

#ifdef _WIN32
#include <windows.h>
#endif

// Configura la consola de Windows para mostrar UTF-8.
// Sin esto, los acentos y la enie salen como caracteres raros
// (fu||tbol en vez de futbol).
void configurar_consola() {
#ifdef _WIN32
    SetConsoleOutputCP(CP_UTF8);
    SetConsoleCP(CP_UTF8);
#endif
}

void mostrar_ayuda() {
    std::cout << "\nComandos:\n";
    std::cout << "  /salir              terminar\n";
    std::cout << "  /temp <n>           temperatura (0 = greedy, 0.7 normal)\n";
    std::cout << "  /penal <n>          penalizacion por repeticion (1.0 = off)\n";
    std::cout << "  /topk <n>           cuantos candidatos considerar\n";
    std::cout << "  /topp <n>           muestreo por nucleo (0.95 tipico)\n";
    std::cout << "  /max <n>            maximo de tokens a generar\n";
    std::cout << "  /sistema <texto>    cambiar el prompt de sistema\n";
    std::cout << "  /config             ver la configuracion actual\n";
    std::cout << "  /ayuda              esta lista\n";
}

int main(int argc, char** argv) {
    configurar_consola();

    std::string ruta = "C:/Users/Said/source/repos/IA_Prueba/IA_Prueba/qwen2.5-1.5b-instruct-q4_0.gguf";
    if (argc >= 2) ruta = argv[1];

    std::cout << "=== Chat con Qwen (motor propio en C++) ===\n\n";

    Qwen modelo;
    try {
        modelo.cargar(ruta);
    }
    catch (const std::exception& e) {
        std::cout << "ERROR: " << e.what() << "\n";
        return 1;
    }

    if (!modelo.bpe.listo()) {
        std::cout << "\nERROR: el tokenizador BPE no se construyo.\n";
        return 1;
    }

    ConfigSampling cfg;
    Sampler sampler;
    int max_tokens = 300;
    std::string sistema = "Eres un asistente util. Respondes de forma clara y concisa.";

    std::cout << "\nEscribe tu mensaje. /ayuda para ver los comandos.\n";
    std::cout << std::string(60, '-') << "\n";

    while (true) {
        std::cout << "\n> " << std::flush;
        std::string entrada;
        if (!std::getline(std::cin, entrada)) break;
        if (entrada.empty()) continue;

        // ---------------- Comandos ----------------
        if (entrada == "/salir" || entrada == "/exit") break;

        if (entrada == "/ayuda") { mostrar_ayuda(); continue; }

        if (entrada == "/config") {
            std::cout << "  temperatura: " << cfg.temperatura << "\n";
            std::cout << "  top_k: " << cfg.top_k << "\n";
            std::cout << "  top_p: " << cfg.top_p << "\n";
            std::cout << "  penalizacion: " << cfg.penal_repeticion << "\n";
            std::cout << "  max tokens: " << max_tokens << "\n";
            std::cout << "  sistema: \"" << sistema << "\"\n";
            continue;
        }

        if (entrada.rfind("/temp ", 0) == 0) {
            cfg.temperatura = std::stof(entrada.substr(6));
            std::cout << "Temperatura: " << cfg.temperatura
                << (cfg.temperatura <= 0 ? " (greedy)" : "") << "\n";
            continue;
        }
        if (entrada.rfind("/penal ", 0) == 0) {
            cfg.penal_repeticion = std::stof(entrada.substr(7));
            std::cout << "Penalizacion: " << cfg.penal_repeticion << "\n";
            continue;
        }
        if (entrada.rfind("/topk ", 0) == 0) {
            cfg.top_k = std::stoi(entrada.substr(6));
            std::cout << "Top-k: " << cfg.top_k << "\n";
            continue;
        }
        if (entrada.rfind("/topp ", 0) == 0) {
            cfg.top_p = std::stof(entrada.substr(6));
            std::cout << "Top-p: " << cfg.top_p << "\n";
            continue;
        }
        if (entrada.rfind("/max ", 0) == 0) {
            max_tokens = std::stoi(entrada.substr(5));
            std::cout << "Max tokens: " << max_tokens << "\n";
            continue;
        }
        if (entrada.rfind("/sistema ", 0) == 0) {
            sistema = entrada.substr(9);
            std::cout << "Prompt de sistema actualizado.\n";
            continue;
        }

        // ---------------- Generar ----------------
        std::vector<int> prompt = modelo.armar_chat(entrada, sistema);

        auto cache = modelo.crear_cache();
        sampler.reiniciar();   // la penalizacion es por respuesta, no global

        auto t1 = std::chrono::high_resolution_clock::now();
        Matrix logits = modelo.procesar_prompt(prompt, cache);

        std::cout << "\n";
        int generados = 0;
        for (int i = 0; i < max_tokens; i++) {
            int sig = sampler.elegir(logits, cfg);

            if (sig == modelo.eos || sig == modelo.bpe.id_de("<|im_end|>"))
                break;

            std::cout << modelo.decodificar_uno(sig) << std::flush;
            generados++;

            if (cache.tokens_procesados() >= modelo.contexto_max - 1) break;
            logits = modelo.forward(sig, cache.tokens_procesados(), cache);
        }

        auto t2 = std::chrono::high_resolution_clock::now();
        double seg = std::chrono::duration<double>(t2 - t1).count();
        std::cout << "\n\n[" << prompt.size() << " tokens de prompt, "
            << generados << " generados, "
            << std::fixed << std::setprecision(1)
            << (generados / seg) << " tokens/s]\n";
    }

    std::cout << "\nHasta luego.\n";
    return 0;
}