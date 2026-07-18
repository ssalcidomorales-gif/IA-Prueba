// test_gguf.cpp
// Construye un archivo GGUF sintetico con valores conocidos y verifica
// que el lector lo interpreta correctamente. No necesita descargar nada.

#include "gguf.h"
#include <iostream>
#include <fstream>
#include <cstring>
#include <cmath>

int ok = 0, total = 0;
void afirmar(const std::string& n, bool c) {
    total++;
    if (c) { std::cout << "[OK]    " << n << "\n"; ok++; }
    else   { std::cout << "[FALLO] " << n << "\n"; }
}
bool casi(float a, float b, float tol=1e-4f){ return std::fabs(a-b)<tol; }

uint16_t fp32_a_fp16(float f) {
    uint32_t bits; std::memcpy(&bits,&f,4);
    uint32_t s=(bits>>16)&0x8000; int32_t e=((bits>>23)&0xFF)-127+15;
    uint32_t m=(bits>>13)&0x3FF;
    if(e<=0) return (uint16_t)s;
    if(e>=31) return (uint16_t)(s|0x7C00);
    return (uint16_t)(s|(e<<10)|m);
}

// --- Helpers para escribir el archivo ---
void esc_str(std::ofstream& f, const std::string& s) {
    uint64_t len = s.size();
    f.write((char*)&len, 8);
    f.write(s.data(), len);
}
template<typename T> void esc(std::ofstream& f, T v) {
    f.write((char*)&v, sizeof(T));
}

void crear_gguf_prueba(const std::string& ruta) {
    std::ofstream f(ruta, std::ios::binary);

    // --- HEADER ---
    f.write("GGUF", 4);
    esc<uint32_t>(f, 3);        // version
    esc<uint64_t>(f, 2);        // 2 tensores
    esc<uint64_t>(f, 5);        // 5 metadatos

    // --- METADATOS ---
    // general.architecture = "prueba" (string)
    esc_str(f, "general.architecture");
    esc<uint32_t>(f, GGUF_STRING);
    esc_str(f, "prueba");

    // prueba.block_count = 12 (uint32)
    esc_str(f, "prueba.block_count");
    esc<uint32_t>(f, GGUF_UINT32);
    esc<uint32_t>(f, 12);

    // prueba.epsilon = 0.00001 (float32)
    esc_str(f, "prueba.epsilon");
    esc<uint32_t>(f, GGUF_FLOAT32);
    esc<float>(f, 1e-5f);

    // tokenizer.vocab = ["hola","mundo","!"] (array de strings)
    esc_str(f, "tokenizer.vocab");
    esc<uint32_t>(f, GGUF_ARRAY);
    esc<uint32_t>(f, GGUF_STRING);
    esc<uint64_t>(f, 3);
    esc_str(f, "hola"); esc_str(f, "mundo"); esc_str(f, "!");

    // prueba.flag = true (bool)
    esc_str(f, "prueba.flag");
    esc<uint32_t>(f, GGUF_BOOL);
    esc<int8_t>(f, 1);

    // --- DESCRIPTORES DE TENSOR ---
    // tensor "pesos_f32": [4, 2] en F32 -> 8 elementos, offset 0
    esc_str(f, "pesos_f32");
    esc<uint32_t>(f, 2);          // 2 dimensiones
    esc<uint64_t>(f, 4);          // dims[0]
    esc<uint64_t>(f, 2);          // dims[1]
    esc<uint32_t>(f, GGML_F32);
    esc<uint64_t>(f, 0);          // offset

    // tensor "pesos_q8": [32] en Q8_0 -> 32 elementos, offset 32
    // (8 floats = 32 bytes, y como 32 ya es multiplo de 32, va justo ahi)
    esc_str(f, "pesos_q8");
    esc<uint32_t>(f, 1);
    esc<uint64_t>(f, 32);
    esc<uint32_t>(f, GGML_Q8_0);
    esc<uint64_t>(f, 32);

    // --- PADDING hasta multiplo de 32 ---
    uint64_t pos = (uint64_t)f.tellp();
    uint64_t pad = (32 - (pos % 32)) % 32;
    for (uint64_t i = 0; i < pad; i++) esc<uint8_t>(f, 0);

    // --- DATOS ---
    // pesos_f32: 1.0, 2.0, ..., 8.0
    for (int i = 1; i <= 8; i++) esc<float>(f, (float)i);

    // pesos_q8: escala 0.5, valores 0..31
    esc<uint16_t>(f, fp32_a_fp16(0.5f));
    for (int i = 0; i < 32; i++) esc<int8_t>(f, (int8_t)i);

    f.close();
}

int main() {
    std::cout << "=== Tests del lector GGUF ===\n\n";

    const std::string ruta = "prueba.gguf";
    crear_gguf_prueba(ruta);

    GGUF g(ruta);

    // --- Header ---
    afirmar("version = 3", g.version == 3);
    afirmar("2 tensores", g.num_tensores == 2);
    afirmar("5 metadatos", g.num_metadatos == 5);
    afirmar("alineacion = 32", g.alineacion == 32);

    // --- Metadatos ---
    afirmar("arquitectura = 'prueba'", g.arquitectura() == "prueba");
    afirmar("block_count = 12", g.entero("prueba.block_count") == 12);
    afirmar("hiper('block_count') = 12", g.hiper("block_count") == 12);
    afirmar("epsilon ~ 1e-5", std::fabs(g.real("prueba.epsilon") - 1e-5) < 1e-9);
    afirmar("flag = true", g.entero("prueba.flag") == 1);

    // --- Arreglo de strings ---
    {
        auto* v = g.arreglo_textos("tokenizer.vocab");
        bool bien = v && v->size()==3 &&
                    (*v)[0]=="hola" && (*v)[1]=="mundo" && (*v)[2]=="!";
        afirmar("vocab = [hola, mundo, !]", bien);
    }

    // --- Clave inexistente devuelve el default ---
    afirmar("clave inexistente -> default", g.entero("no.existe", 99) == 99);

    // --- Descriptores de tensor ---
    {
        afirmar("encuentra 'pesos_f32'", g.tiene_tensor("pesos_f32"));
        afirmar("encuentra 'pesos_q8'",  g.tiene_tensor("pesos_q8"));
        const auto& t = g.tensores.at("pesos_f32");
        afirmar("pesos_f32 tiene 8 elementos", t.num_elementos() == 8);
        afirmar("pesos_f32 es F32", t.tipo == GGML_F32);
    }

    // --- Leer tensor F32 ---
    {
        Matrix m = g.leer_tensor("pesos_f32");
        bool bien = (m.rows==2 && m.cols==4);
        for (int i = 0; i < 8; i++)
            if (!casi(m.data[i], (float)(i+1))) bien = false;
        afirmar("leer_tensor F32: valores 1..8 correctos", bien);
    }

    // --- Leer tensor Q8_0 (dequantiza al vuelo) ---
    {
        Matrix m = g.leer_tensor("pesos_q8");
        bool bien = (m.rows==1 && m.cols==32);
        for (int i = 0; i < 32; i++)
            if (!casi(m.data[i], i * 0.5f)) bien = false;
        afirmar("leer_tensor Q8_0: dequantiza correctamente", bien);
    }

    // --- Tensor inexistente lanza error ---
    {
        bool lanzo = false;
        try { g.leer_tensor("no_existe"); }
        catch (const std::exception&) { lanzo = true; }
        afirmar("tensor inexistente lanza error", lanzo);
    }

    // --- Archivo que no es GGUF ---
    {
        std::ofstream basura("basura.bin", std::ios::binary);
        basura << "esto no es un gguf para nada";
        basura.close();
        bool lanzo = false;
        try { GGUF g2("basura.bin"); }
        catch (const std::exception&) { lanzo = true; }
        afirmar("archivo no-GGUF es rechazado", lanzo);
        std::remove("basura.bin");
    }

    std::remove(ruta.c_str());

    std::cout << "\n=== " << ok << "/" << total << " pruebas pasaron ===\n";
    return (ok == total) ? 0 : 1;
}
