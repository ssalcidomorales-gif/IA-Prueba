// safetensors.h
// Lector de archivos .safetensors en C++ puro. Sin librerias externas.
//
// Formato del archivo:
//   [8 bytes] N  = tamano del header JSON (entero de 64 bits, little-endian)
//   [N bytes] header JSON (UTF-8), describe cada tensor:
//             nombre -> { dtype, shape, data_offsets: [inicio, fin] }
//   [resto]   datos crudos de todos los tensores, uno tras otro
//
// Los data_offsets son relativos al INICIO del bloque de datos,
// que empieza en el byte (8 + N). Asi que un tensor vive en el archivo
// desde el byte (8 + N + inicio) hasta (8 + N + fin).
//
// Solo manejamos dtype F32 (float32), que es lo que trae GPT-2 original.

#ifndef SAFETENSORS_H
#define SAFETENSORS_H

#include "matrix.h"
#include <fstream>
#include <string>
#include <map>
#include <vector>
#include <cstdint>
#include <stdexcept>
#include <iostream>

// Informacion de un tensor tal como aparece en el header
struct InfoTensor {
    std::string dtype;            // "F32", "F16", etc. Nosotros solo usamos F32.
    std::vector<int> shape;       // dimensiones, p.ej. {50257, 768}
    uint64_t offset_inicio;       // byte de inicio dentro del bloque de datos
    uint64_t offset_fin;          // byte final (exclusivo)
};

class SafeTensors {
public:
    // Abre el archivo y lee/parsea el header. No carga los datos todavia
    // (serian cientos de MB); los tensores se leen bajo demanda con getTensor.
    explicit SafeTensors(const std::string& ruta) : ruta_(ruta) {
        archivo_.open(ruta, std::ios::binary);
        if (!archivo_)
            throw std::runtime_error("No pude abrir el archivo: " + ruta);

        // 1. Leer los primeros 8 bytes -> tamano del header
        uint64_t tam_header = 0;
        archivo_.read(reinterpret_cast<char*>(&tam_header), 8);
        if (!archivo_)
            throw std::runtime_error("Archivo demasiado corto para el header");

        // 2. Leer los N bytes del header JSON
        std::string json(tam_header, '\0');
        archivo_.read(&json[0], tam_header);

        // 3. El bloque de datos empieza justo despues
        inicio_datos_ = 8 + tam_header;

        // 4. Parsear el JSON (mini-parser propio, abajo)
        parsearHeader(json);
    }

    // Lista los nombres de todos los tensores disponibles
    std::vector<std::string> nombres() const {
        std::vector<std::string> res;
        for (const auto& par : tensores_) res.push_back(par.first);
        return res;
    }

    // Devuelve la info (forma, dtype, offsets) de un tensor por nombre
    const InfoTensor& info(const std::string& nombre) const {
        auto it = tensores_.find(nombre);
        if (it == tensores_.end())
            throw std::runtime_error("No existe el tensor: " + nombre);
        return it->second;
    }

    // Carga un tensor 2D como Matrix. Si el tensor es 1D (un vector, como los
    // bias), lo devuelve como una Matrix de [1 x N].
    Matrix getTensor(const std::string& nombre) {
        const InfoTensor& t = info(nombre);

        if (t.dtype != "F32")
            throw std::runtime_error("Solo soporto F32 por ahora. " + nombre +
                                     " es " + t.dtype);

        // Calcular filas y columnas
        int filas, cols;
        if (t.shape.size() == 1) {
            filas = 1;
            cols = t.shape[0];
        } else if (t.shape.size() == 2) {
            filas = t.shape[0];
            cols = t.shape[1];
        } else {
            throw std::runtime_error("Solo manejo tensores 1D o 2D. " + nombre +
                                     " tiene " + std::to_string(t.shape.size()) +
                                     " dimensiones");
        }

        // Cuantos floats son
        uint64_t bytes = t.offset_fin - t.offset_inicio;
        uint64_t n_floats = bytes / 4;  // cada float32 son 4 bytes

        if ((int)n_floats != filas * cols)
            throw std::runtime_error("La forma no cuadra con los bytes en " + nombre);

        // Ir a la posicion del tensor en el archivo y leer los floats
        Matrix m(filas, cols);
        archivo_.clear();  // limpiar posibles flags de EOF de lecturas previas
        archivo_.seekg(inicio_datos_ + t.offset_inicio, std::ios::beg);
        archivo_.read(reinterpret_cast<char*>(m.data.data()), bytes);
        if (!archivo_)
            throw std::runtime_error("Error leyendo los datos de " + nombre);

        return m;
    }

private:
    std::string ruta_;
    std::ifstream archivo_;
    uint64_t inicio_datos_ = 0;
    std::map<std::string, InfoTensor> tensores_;

    // --- Mini-parser de JSON, hecho a mano ---
    // El header de safetensors es JSON plano y predecible, asi que en vez de
    // meter una libreria, escaneamos el texto buscando los campos que nos
    // importan. No es un parser general de JSON: aprovecha que la estructura
    // siempre es la misma. Suficiente y sin dependencias.
    void parsearHeader(const std::string& json) {
        size_t i = 0;
        size_t n = json.size();

        // Saltar el '{' inicial
        while (i < n && json[i] != '{') i++;
        i++;

        while (i < n) {
            // Buscar la siguiente clave entre comillas
            size_t comilla1 = json.find('"', i);
            if (comilla1 == std::string::npos) break;
            size_t comilla2 = json.find('"', comilla1 + 1);
            if (comilla2 == std::string::npos) break;

            std::string clave = json.substr(comilla1 + 1, comilla2 - comilla1 - 1);
            i = comilla2 + 1;

            // El campo especial "__metadata__" no es un tensor; saltarlo
            if (clave == "__metadata__") {
                i = saltarObjeto(json, i);
                continue;
            }

            // Ahora viene ':' y luego el objeto { ... } de este tensor
            size_t llave = json.find('{', i);
            if (llave == std::string::npos) break;
            size_t fin_obj = encontrarCierre(json, llave);
            std::string obj = json.substr(llave, fin_obj - llave + 1);

            // Extraer los campos de dentro del objeto
            InfoTensor info;
            info.dtype = extraerStringTrasClave(obj, "dtype");
            info.shape = extraerArregloEnteros(obj, "shape");
            std::vector<int> offs = extraerArregloEnteros(obj, "data_offsets");
            if (offs.size() == 2) {
                info.offset_inicio = (uint64_t)offs[0];
                info.offset_fin = (uint64_t)offs[1];
            }

            tensores_[clave] = info;

            i = fin_obj + 1;

            // Si sigue una coma vamos por el siguiente; si es '}' terminamos
            size_t siguiente = json.find_first_not_of(" \t\n\r", i);
            if (siguiente == std::string::npos) break;
            if (json[siguiente] == '}') break;
        }
    }

    // Dada la posicion de un '{', encuentra su '}' correspondiente
    // (contando anidamiento).
    size_t encontrarCierre(const std::string& s, size_t abre) {
        int profundidad = 0;
        for (size_t k = abre; k < s.size(); k++) {
            if (s[k] == '{') profundidad++;
            else if (s[k] == '}') {
                profundidad--;
                if (profundidad == 0) return k;
            }
        }
        throw std::runtime_error("JSON mal formado: llave sin cerrar");
    }

    // Salta un objeto { ... } completo y devuelve la posicion despues del '}'
    size_t saltarObjeto(const std::string& s, size_t desde) {
        size_t llave = s.find('{', desde);
        if (llave == std::string::npos) return s.size();
        return encontrarCierre(s, llave) + 1;
    }

    // Busca "clave":"valor" dentro de obj y devuelve el valor string
    std::string extraerStringTrasClave(const std::string& obj,
                                       const std::string& clave) {
        std::string patron = "\"" + clave + "\"";
        size_t p = obj.find(patron);
        if (p == std::string::npos) return "";
        p = obj.find(':', p);
        size_t c1 = obj.find('"', p);
        size_t c2 = obj.find('"', c1 + 1);
        return obj.substr(c1 + 1, c2 - c1 - 1);
    }

    // Busca "clave":[a, b, c] dentro de obj y devuelve {a, b, c}
    std::vector<int> extraerArregloEnteros(const std::string& obj,
                                           const std::string& clave) {
        std::vector<int> res;
        std::string patron = "\"" + clave + "\"";
        size_t p = obj.find(patron);
        if (p == std::string::npos) return res;
        size_t abre = obj.find('[', p);
        size_t cierra = obj.find(']', abre);
        std::string dentro = obj.substr(abre + 1, cierra - abre - 1);

        // Separar por comas y convertir a enteros
        std::string num;
        for (char c : dentro) {
            if (c == ',' ) {
                if (!num.empty()) { res.push_back(std::stoi(num)); num.clear(); }
            } else if (!isspace((unsigned char)c)) {
                num += c;
            }
        }
        if (!num.empty()) res.push_back(std::stoi(num));
        return res;
    }
};

#endif // SAFETENSORS_H
