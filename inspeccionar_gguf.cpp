// inspeccionar_gguf.cpp
// Abre un archivo GGUF real y muestra todo lo que contiene: metadatos,
// arquitectura, hiperparametros, y la lista de tensores con su tipo
// de cuantizacion y tamano.
//
// Correr:  inspeccionar_gguf  C:/ruta/modelo.gguf

#include "gguf.h"
#include <iostream>
#include <iomanip>
#include <map>
#include <algorithm>

std::string formatear_bytes(uint64_t b) {
    const char* u[] = {"B", "KB", "MB", "GB"};
    double v = (double)b;
    int i = 0;
    while (v >= 1024.0 && i < 3) { v /= 1024.0; i++; }
    char buf[64];
    snprintf(buf, sizeof(buf), "%.2f %s", v, u[i]);
    return buf;
}

// Recorta un texto largo para que no reviente la pantalla
std::string recortar(const std::string& s, size_t max = 60) {
    if (s.size() <= max) return s;
    return s.substr(0, max) + "... (" + std::to_string(s.size()) + " chars)";
}

std::string valor_a_texto(const GGUFValor& v) {
    if (v.es_arreglo()) {
        size_t n = v.arr_textos.size() + v.arr_enteros.size() + v.arr_reales.size();
        std::string s = "[" + std::to_string(n) + " elementos";
        // Mostrar los primeros para dar idea del contenido
        if (!v.arr_textos.empty()) {
            s += ": ";
            for (size_t i = 0; i < std::min<size_t>(3, v.arr_textos.size()); i++) {
                s += "\"" + recortar(v.arr_textos[i], 20) + "\"";
                if (i + 1 < std::min<size_t>(3, v.arr_textos.size())) s += ", ";
            }
            if (v.arr_textos.size() > 3) s += ", ...";
        } else if (!v.arr_enteros.empty()) {
            s += ": ";
            for (size_t i = 0; i < std::min<size_t>(5, v.arr_enteros.size()); i++) {
                s += std::to_string(v.arr_enteros[i]);
                if (i + 1 < std::min<size_t>(5, v.arr_enteros.size())) s += ", ";
            }
            if (v.arr_enteros.size() > 5) s += ", ...";
        }
        return s + "]";
    }
    switch (v.tipo) {
        case GGUF_STRING:  return "\"" + recortar(v.texto) + "\"";
        case GGUF_FLOAT32:
        case GGUF_FLOAT64: return std::to_string(v.real);
        case GGUF_BOOL:    return v.entero ? "true" : "false";
        default:           return std::to_string(v.entero);
    }
}

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cout << "Uso: inspeccionar_gguf <ruta_del_modelo.gguf>\n\n";
        std::cout << "Ejemplo:\n";
        std::cout << "  inspeccionar_gguf C:/Modelos/qwen2.5-0.5b-q4_0.gguf\n";
        return 1;
    }

    std::string ruta = argv[1];
    std::cout << "Abriendo: " << ruta << "\n\n";

    try {
        GGUF g(ruta);

        // ============ RESUMEN ============
        std::cout << "==================== RESUMEN ====================\n";
        std::cout << "  Version GGUF:   " << g.version << "\n";
        std::cout << "  Arquitectura:   " << g.arquitectura() << "\n";
        std::cout << "  Tensores:       " << g.num_tensores << "\n";
        std::cout << "  Metadatos:      " << g.num_metadatos << "\n";
        std::cout << "  Alineacion:     " << g.alineacion << " bytes\n";
        std::cout << "  Datos empiezan: byte " << g.inicio_datos << "\n";

        // ============ HIPERPARAMETROS ============
        std::cout << "\n============== HIPERPARAMETROS ==============\n";
        const char* claves[] = {
            "block_count", "context_length", "embedding_length",
            "feed_forward_length", "attention.head_count",
            "attention.head_count_kv", "rope.freq_base",
            "attention.layer_norm_rms_epsilon"
        };
        for (const char* c : claves) {
            std::string clave = g.arquitectura() + std::string(".") + c;
            if (g.tiene(clave)) {
                auto& v = g.meta.at(clave);
                std::cout << "  " << std::left << std::setw(36) << c
                          << valor_a_texto(v) << "\n";
            }
        }

        // ============ TODOS LOS METADATOS ============
        std::cout << "\n============== METADATOS COMPLETOS ==============\n";
        for (const auto& par : g.meta) {
            std::cout << "  " << std::left << std::setw(42) << par.first
                      << valor_a_texto(par.second) << "\n";
        }

        // ============ TENSORES ============
        std::cout << "\n================= TENSORES =================\n";
        std::cout << std::left << std::setw(44) << "NOMBRE"
                  << std::setw(8)  << "TIPO"
                  << std::setw(22) << "FORMA"
                  << "TAMANO\n";
        std::cout << std::string(90, '-') << "\n";

        uint64_t bytes_totales = 0;
        uint64_t params_totales = 0;
        std::map<std::string, int> conteo_tipos;
        std::map<std::string, uint64_t> bytes_por_tipo;

        // Mostrar en el orden en que aparecen en el archivo
        int mostrados = 0;
        const int MAX_MOSTRAR = 40;
        for (const auto& nombre : g.orden_tensores) {
            const GGUFTensor& t = g.tensores.at(nombre);
            size_t n = t.num_elementos();
            size_t bytes = bytes_del_tensor(t.tipo, n);

            bytes_totales  += bytes;
            params_totales += n;
            conteo_tipos[nombre_tipo(t.tipo)]++;
            bytes_por_tipo[nombre_tipo(t.tipo)] += bytes;

            if (mostrados < MAX_MOSTRAR) {
                std::cout << std::left << std::setw(44) << recortar(nombre, 42)
                          << std::setw(8)  << nombre_tipo(t.tipo)
                          << std::setw(22) << t.forma_texto()
                          << formatear_bytes(bytes) << "\n";
                mostrados++;
            }
        }
        if (g.orden_tensores.size() > MAX_MOSTRAR)
            std::cout << "  ... y " << (g.orden_tensores.size() - MAX_MOSTRAR)
                      << " tensores mas\n";

        // ============ ESTADISTICAS ============
        std::cout << "\n================ ESTADISTICAS ================\n";
        std::cout << "  Parametros totales: " << params_totales
                  << "  (~" << (params_totales / 1000000) << "M)\n";
        std::cout << "  Tamano de los pesos: " << formatear_bytes(bytes_totales) << "\n";

        if (params_totales > 0) {
            double bits = (double)bytes_totales * 8.0 / (double)params_totales;
            std::cout << "  Bits por peso: " << std::fixed << std::setprecision(2)
                      << bits << "\n";
            uint64_t si_f32 = params_totales * 4;
            std::cout << "  En float32 pesaria: " << formatear_bytes(si_f32)
                      << "  (compresion " << std::setprecision(2)
                      << ((double)si_f32 / bytes_totales) << "x)\n";
        }

        std::cout << "\n  Tipos de cuantizacion usados:\n";
        for (const auto& par : conteo_tipos) {
            std::cout << "    " << std::left << std::setw(8) << par.first
                      << std::setw(6) << par.second << " tensores, "
                      << formatear_bytes(bytes_por_tipo[par.first]) << "\n";
        }

        // ============ TOKENIZER ============
        std::cout << "\n================= TOKENIZER =================\n";
        if (auto* v = g.arreglo_textos("tokenizer.ggml.tokens")) {
            std::cout << "  Vocabulario: " << v->size() << " tokens\n";
            std::cout << "  Primeros: ";
            for (size_t i = 0; i < std::min<size_t>(8, v->size()); i++)
                std::cout << "\"" << recortar((*v)[i], 12) << "\" ";
            std::cout << "\n";
        } else {
            std::cout << "  (sin vocabulario embebido)\n";
        }
        if (g.tiene("tokenizer.ggml.model"))
            std::cout << "  Modelo de tokenizer: "
                      << g.texto("tokenizer.ggml.model") << "\n";

        std::cout << "\nListo. El lector GGUF funciona con este modelo.\n";

    } catch (const std::exception& e) {
        std::cout << "ERROR: " << e.what() << "\n\n";
        std::cout << "Revisa que la ruta sea correcta y que el archivo\n";
        std::cout << "sea un GGUF completo (no un pointer file de Git LFS).\n";
        return 1;
    }

    return 0;
}
