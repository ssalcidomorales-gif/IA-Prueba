// gguf.h
// FASE 2 - Lector del formato GGUF.
//
// GGUF es el formato de llama.cpp para modelos cuantizados. A diferencia
// de safetensors (que solo guarda tensores), GGUF trae metadatos ricos:
// arquitectura, hiperparametros, vocabulario completo, plantilla de chat.
// Todo lo necesario para correr el modelo va en un solo archivo.
//
// ESTRUCTURA DEL ARCHIVO
//
//   +-------------------------+
//   | HEADER (24 bytes)       |  magic "GGUF" + version + conteos
//   +-------------------------+
//   | METADATOS               |  pares clave-valor tipados
//   +-------------------------+
//   | DESCRIPTORES DE TENSOR  |  nombre, dimensiones, tipo, offset
//   +-------------------------+
//   | PADDING                 |  hasta multiplo de 32 bytes
//   +-------------------------+
//   | DATOS DE LOS TENSORES   |  los bytes crudos (posiblemente cuantizados)
//   +-------------------------+
//
// DETALLES QUE IMPORTAN
//   - Todo es little-endian por defecto
//   - Los strings son: uint64 con la longitud + bytes UTF-8, SIN terminador
//   - Las dimensiones van en ORDEN INVERSO: un tensor [4096, 11008] se
//     guarda como [11008, 4096]. Esto viene de la convencion de ggml.
//   - Los offsets de tensor son relativos al inicio de la seccion de datos,
//     no al inicio del archivo
//   - La alineacion por defecto es 32, pero puede cambiarla la clave
//     de metadatos "general.alignment"

#ifndef GGUF_H
#define GGUF_H

#include "cuantizacion.h"
#include "matrix.h"
#include <fstream>
#include <string>
#include <vector>
#include <map>
#include <cstdint>
#include <stdexcept>
#include <iostream>

// ----------------------------------------------------------------------
// TIPOS DE VALOR EN LOS METADATOS
// ----------------------------------------------------------------------
enum GGUFTipo : uint32_t {
    GGUF_UINT8   = 0,
    GGUF_INT8    = 1,
    GGUF_UINT16  = 2,
    GGUF_INT16   = 3,
    GGUF_UINT32  = 4,
    GGUF_INT32   = 5,
    GGUF_FLOAT32 = 6,
    GGUF_BOOL    = 7,
    GGUF_STRING  = 8,
    GGUF_ARRAY   = 9,
    GGUF_UINT64  = 10,
    GGUF_INT64   = 11,
    GGUF_FLOAT64 = 12,
};

// Un valor de metadatos. Puede ser escalar, string o arreglo.
struct GGUFValor {
    uint32_t tipo = 0;

    // Escalares (solo uno es valido segun 'tipo')
    int64_t  entero = 0;
    double   real = 0.0;
    std::string texto;

    // Arreglos
    uint32_t tipo_elemento = 0;
    std::vector<int64_t>     arr_enteros;
    std::vector<double>      arr_reales;
    std::vector<std::string> arr_textos;

    bool es_arreglo() const { return tipo == GGUF_ARRAY; }

    // Accesos convenientes con valor por defecto
    int64_t como_entero(int64_t def = 0) const {
        return (tipo == GGUF_ARRAY || tipo == GGUF_STRING) ? def : entero;
    }
    double como_real(double def = 0.0) const {
        return (tipo == GGUF_FLOAT32 || tipo == GGUF_FLOAT64) ? real : def;
    }
    std::string como_texto(const std::string& def = "") const {
        return (tipo == GGUF_STRING) ? texto : def;
    }
};

// Descriptor de un tensor: donde esta y como interpretarlo
struct GGUFTensor {
    std::string nombre;
    std::vector<uint64_t> dims;   // ya en orden natural (invertidas al leer)
    uint32_t tipo = 0;            // GGMLType
    uint64_t offset = 0;          // relativo al inicio de la seccion de datos

    size_t num_elementos() const {
        size_t n = 1;
        for (uint64_t d : dims) n *= (size_t)d;
        return n;
    }

    std::string forma_texto() const {
        std::string s = "[";
        for (size_t i = 0; i < dims.size(); i++) {
            s += std::to_string(dims[i]);
            if (i + 1 < dims.size()) s += ", ";
        }
        return s + "]";
    }
};

// ----------------------------------------------------------------------
// LECTOR
// ----------------------------------------------------------------------
class GGUF {
public:
    uint32_t version = 0;
    uint64_t num_tensores = 0;
    uint64_t num_metadatos = 0;
    uint64_t alineacion = 32;
    uint64_t inicio_datos = 0;   // offset absoluto donde empiezan los tensores

    std::map<std::string, GGUFValor> meta;
    std::map<std::string, GGUFTensor> tensores;
    std::vector<std::string> orden_tensores;  // para listar en orden de archivo

    explicit GGUF(const std::string& ruta) : ruta_(ruta) {
        archivo_.open(ruta, std::ios::binary);
        if (!archivo_)
            throw std::runtime_error("No pude abrir el archivo: " + ruta);

        leer_header();
        leer_metadatos();
        leer_descriptores();
        calcular_inicio_datos();
    }

    // ------------------------------------------------------------------
    // Lectura de tensores
    // ------------------------------------------------------------------

    // Carga un tensor y lo dequantiza a float32.
    // Devuelve una Matrix [filas x columnas].
    //
    // OJO CON LAS DIMENSIONES: en ggml, dims[0] es la dimension que varia
    // mas rapido en memoria (la "interna"). Para una matriz de pesos
    // [salida, entrada], dims[0] = entrada y dims[1] = salida. Como
    // nuestra Matrix es row-major con filas x cols, mapeamos:
    //   filas = dims[1] (o 1 si es un vector)
    //   cols  = dims[0]
    Matrix leer_tensor(const std::string& nombre) {
        auto it = tensores.find(nombre);
        if (it == tensores.end())
            throw std::runtime_error("No existe el tensor: " + nombre);

        const GGUFTensor& t = it->second;
        size_t n = t.num_elementos();

        // Leer los bytes crudos del archivo
        size_t bytes = bytes_del_tensor(t.tipo, n);
        if (bytes == 0)
            throw std::runtime_error(
                "Tipo no soportado en '" + nombre + "': " + nombre_tipo(t.tipo));

        std::vector<uint8_t> crudo(bytes);
        archivo_.clear();
        archivo_.seekg(inicio_datos + t.offset, std::ios::beg);
        archivo_.read((char*)crudo.data(), bytes);
        if (!archivo_)
            throw std::runtime_error("Error leyendo los datos de: " + nombre);

        // Dequantizar a float32
        std::vector<float> valores = dequantizar(crudo.data(), t.tipo, n);

        // Armar la Matrix
        int cols  = t.dims.empty() ? 1 : (int)t.dims[0];
        int filas = (t.dims.size() >= 2) ? (int)t.dims[1] : 1;

        Matrix m(filas, cols);
        std::memcpy(m.data.data(), valores.data(), n * sizeof(float));
        return m;
    }

    bool tiene_tensor(const std::string& nombre) const {
        return tensores.count(nombre) > 0;
    }

    // ------------------------------------------------------------------
    // Consulta de metadatos
    // ------------------------------------------------------------------
    bool tiene(const std::string& clave) const { return meta.count(clave) > 0; }

    int64_t entero(const std::string& clave, int64_t def = 0) const {
        auto it = meta.find(clave);
        return (it == meta.end()) ? def : it->second.como_entero(def);
    }

    double real(const std::string& clave, double def = 0.0) const {
        auto it = meta.find(clave);
        return (it == meta.end()) ? def : it->second.como_real(def);
    }

    std::string texto(const std::string& clave,
                      const std::string& def = "") const {
        auto it = meta.find(clave);
        return (it == meta.end()) ? def : it->second.como_texto(def);
    }

    const std::vector<std::string>* arreglo_textos(
            const std::string& clave) const {
        auto it = meta.find(clave);
        if (it == meta.end() || !it->second.es_arreglo()) return nullptr;
        return &it->second.arr_textos;
    }

    const std::vector<int64_t>* arreglo_enteros(
            const std::string& clave) const {
        auto it = meta.find(clave);
        if (it == meta.end() || !it->second.es_arreglo()) return nullptr;
        return &it->second.arr_enteros;
    }

    // Arquitectura del modelo: "llama", "qwen2", "gpt2", etc.
    std::string arquitectura() const { return texto("general.architecture"); }

    // Los hiperparametros usan el nombre de la arquitectura como prefijo:
    //   qwen2.block_count, qwen2.attention.head_count, etc.
    int64_t hiper(const std::string& sufijo, int64_t def = 0) const {
        return entero(arquitectura() + "." + sufijo, def);
    }

private:
    std::string ruta_;
    std::ifstream archivo_;

    // --- Lectura de tipos primitivos (little-endian) ---
    template <typename T>
    T leer() {
        T v{};
        archivo_.read((char*)&v, sizeof(T));
        if (!archivo_) throw std::runtime_error("Fin de archivo inesperado");
        return v;
    }

    // Strings: uint64 de longitud + bytes UTF-8, sin terminador
    std::string leer_string() {
        uint64_t len = leer<uint64_t>();
        if (len > 100000000ULL)
            throw std::runtime_error("String absurdamente largo, archivo corrupto");
        std::string s(len, '\0');
        if (len > 0) archivo_.read(&s[0], len);
        return s;
    }

    void leer_header() {
        char magic[4];
        archivo_.read(magic, 4);
        if (std::memcmp(magic, "GGUF", 4) != 0)
            throw std::runtime_error(
                "No es un archivo GGUF (magic incorrecto). "
                "Los primeros 4 bytes deberian ser 'GGUF'.");

        version = leer<uint32_t>();
        if (version < 2 || version > 3)
            std::cerr << "Aviso: version GGUF " << version
                      << " (esperada 2 o 3). Intentando de todos modos.\n";

        num_tensores  = leer<uint64_t>();
        num_metadatos = leer<uint64_t>();
    }

    // Lee un valor segun su tipo. Se usa tanto para escalares sueltos
    // como para los elementos de un arreglo.
    void leer_valor_en(GGUFValor& v, uint32_t tipo) {
        switch (tipo) {
            case GGUF_UINT8:   v.entero = leer<uint8_t>();  break;
            case GGUF_INT8:    v.entero = leer<int8_t>();   break;
            case GGUF_UINT16:  v.entero = leer<uint16_t>(); break;
            case GGUF_INT16:   v.entero = leer<int16_t>();  break;
            case GGUF_UINT32:  v.entero = leer<uint32_t>(); break;
            case GGUF_INT32:   v.entero = leer<int32_t>();  break;
            case GGUF_UINT64:  v.entero = (int64_t)leer<uint64_t>(); break;
            case GGUF_INT64:   v.entero = leer<int64_t>();  break;
            case GGUF_BOOL:    v.entero = leer<int8_t>();   break;
            case GGUF_FLOAT32: v.real   = leer<float>();    break;
            case GGUF_FLOAT64: v.real   = leer<double>();   break;
            case GGUF_STRING:  v.texto  = leer_string();    break;
            default:
                throw std::runtime_error(
                    "Tipo de metadato desconocido: " + std::to_string(tipo));
        }
    }

    void leer_metadatos() {
        for (uint64_t i = 0; i < num_metadatos; i++) {
            std::string clave = leer_string();
            uint32_t tipo = leer<uint32_t>();

            GGUFValor v;
            v.tipo = tipo;

            if (tipo == GGUF_ARRAY) {
                v.tipo_elemento = leer<uint32_t>();
                uint64_t cuantos = leer<uint64_t>();

                for (uint64_t k = 0; k < cuantos; k++) {
                    GGUFValor elem;
                    leer_valor_en(elem, v.tipo_elemento);
                    switch (v.tipo_elemento) {
                        case GGUF_STRING:
                            v.arr_textos.push_back(elem.texto);
                            break;
                        case GGUF_FLOAT32:
                        case GGUF_FLOAT64:
                            v.arr_reales.push_back(elem.real);
                            break;
                        default:
                            v.arr_enteros.push_back(elem.entero);
                            break;
                    }
                }
            } else {
                leer_valor_en(v, tipo);
            }

            // La alineacion puede venir sobreescrita en los metadatos
            if (clave == "general.alignment")
                alineacion = (uint64_t)v.entero;

            meta[clave] = std::move(v);
        }
    }

    void leer_descriptores() {
        for (uint64_t i = 0; i < num_tensores; i++) {
            GGUFTensor t;
            t.nombre = leer_string();

            uint32_t n_dims = leer<uint32_t>();
            if (n_dims > 4)
                throw std::runtime_error(
                    "Tensor con " + std::to_string(n_dims) +
                    " dimensiones (maximo 4)");

            // Las dimensiones vienen en orden inverso al natural
            std::vector<uint64_t> dims_invertidas(n_dims);
            for (uint32_t d = 0; d < n_dims; d++)
                dims_invertidas[d] = leer<uint64_t>();

            t.dims = dims_invertidas;   // las dejamos como vienen de ggml
            t.tipo   = leer<uint32_t>();
            t.offset = leer<uint64_t>();

            orden_tensores.push_back(t.nombre);
            tensores[t.nombre] = std::move(t);
        }
    }

    void calcular_inicio_datos() {
        // Donde quedo el cursor tras leer los descriptores
        uint64_t pos = (uint64_t)archivo_.tellg();
        // Redondear hacia arriba al siguiente multiplo de la alineacion
        inicio_datos = pos + (alineacion - (pos % alineacion)) % alineacion;
    }
};

#endif // GGUF_H
