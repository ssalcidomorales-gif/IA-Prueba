// generacion.h
// Sampling (elegir el siguiente token) y decodificacion (token ID -> texto).
//
// Para DECODIFICAR necesitamos vocab.json de GPT-2, que mapea cada string
// de token a su ID. Lo invertimos: ID -> string. Pero los strings usan el
// "disfraz" byte-level de GPT-2 (el espacio es 'Ġ', etc.), asi que hay que
// deshacer ese mapeo para obtener el texto real.

#ifndef GENERACION_H
#define GENERACION_H

#include "matrix.h"
#include <string>
#include <vector>
#include <map>
#include <fstream>
#include <sstream>
#include <random>
#include <algorithm>
#include <cmath>

// ----------------------------------------------------------------------
// bytes_to_unicode de GPT-2 (el "disfraz")
// ----------------------------------------------------------------------
// Devuelve el mapeo inverso: de codepoint unicode "disfrazado" -> byte real.
// Los bytes imprimibles se mapean a si mismos; los problematicos (espacios,
// control) se desplazan a partir de 256.
std::map<int, int> construir_byte_decoder() {
    std::map<int, int> byte_encoder;  // byte real -> codepoint disfrazado
    std::vector<int> bs;

    // rangos de bytes que se representan a si mismos
    for (int i = (int)'!'; i <= (int)'~'; i++) bs.push_back(i);
    for (int i = 0xA1; i <= 0xAC; i++) bs.push_back(i);
    for (int i = 0xAE; i <= 0xFF; i++) bs.push_back(i);

    std::vector<int> cs = bs;  // copia
    int n = 0;
    // los bytes que NO estan en la lista se mapean a 256+n
    for (int b = 0; b < 256; b++) {
        if (std::find(bs.begin(), bs.end(), b) == bs.end()) {
            bs.push_back(b);
            cs.push_back(256 + n);
            n++;
        }
    }

    // byte_encoder: byte -> codepoint disfrazado
    for (size_t i = 0; i < bs.size(); i++)
        byte_encoder[bs[i]] = cs[i];

    // lo que queremos es el inverso: codepoint disfrazado -> byte
    std::map<int, int> byte_decoder;
    for (auto& par : byte_encoder)
        byte_decoder[par.second] = par.first;
    return byte_decoder;
}

// ----------------------------------------------------------------------
// TOKENIZER (solo decodificacion por ahora)
// ----------------------------------------------------------------------
class Tokenizer {
public:
    std::map<int, std::string> id_a_token;  // ID -> string disfrazado
    std::map<int, int> byte_decoder;        // codepoint disfrazado -> byte

    // Carga vocab.json (formato: {"token_disfrazado": id, ...})
    void cargar_vocab(const std::string& ruta) {
        byte_decoder = construir_byte_decoder();

        std::ifstream f(ruta);
        if (!f) throw std::runtime_error("No pude abrir vocab.json: " + ruta);
        std::stringstream ss;
        ss << f.rdbuf();
        std::string json = ss.str();

        // Parser minimo para {"clave": valor, ...}. Las claves pueden tener
        // caracteres escapados; manejamos \" \\ \u.
        parsear_vocab(json);
    }

    // Convierte una lista de token IDs a texto real
    std::string decodificar(const std::vector<int>& tokens) {
        // 1. concatenar los strings disfrazados de cada token
        std::string disfrazado;
        for (int id : tokens) {
            auto it = id_a_token.find(id);
            if (it != id_a_token.end()) disfrazado += it->second;
        }
        // 2. deshacer el disfraz: cada codepoint -> byte real -> UTF-8
        return deshacer_disfraz(disfrazado);
    }

    // Decodifica un solo token
    std::string decodificar_uno(int id) {
        return decodificar({id});
    }

private:
    // Convierte el string disfrazado (secuencia de codepoints unicode) de
    // vuelta a los bytes reales, y los interpreta como UTF-8.
    std::string deshacer_disfraz(const std::string& s) {
        // s esta en UTF-8; hay que leer sus codepoints uno por uno
        std::vector<int> codepoints = utf8_a_codepoints(s);
        std::string bytes;
        for (int cp : codepoints) {
            auto it = byte_decoder.find(cp);
            if (it != byte_decoder.end())
                bytes += (char)(unsigned char)it->second;
            else
                bytes += (char)(unsigned char)cp;  // fallback
        }
        return bytes;  // ya son los bytes UTF-8 reales del texto
    }

    // Lee un string UTF-8 y devuelve la lista de codepoints
    std::vector<int> utf8_a_codepoints(const std::string& s) {
        std::vector<int> cps;
        size_t i = 0;
        while (i < s.size()) {
            unsigned char c = s[i];
            int cp, len;
            if (c < 0x80)      { cp = c; len = 1; }
            else if (c < 0xE0) { cp = c & 0x1F; len = 2; }
            else if (c < 0xF0) { cp = c & 0x0F; len = 3; }
            else               { cp = c & 0x07; len = 4; }
            for (int k = 1; k < len && i + k < s.size(); k++)
                cp = (cp << 6) | (s[i + k] & 0x3F);
            cps.push_back(cp);
            i += len;
        }
        return cps;
    }

    // Parser del vocab.json
    void parsear_vocab(const std::string& json) {
        size_t i = 0;
        while (i < json.size()) {
            // buscar apertura de clave
            size_t c1 = json.find('"', i);
            if (c1 == std::string::npos) break;
            // leer la clave respetando escapes
            std::string clave;
            size_t k = c1 + 1;
            while (k < json.size() && json[k] != '"') {
                if (json[k] == '\\') {
                    char esc = json[k+1];
                    if (esc == 'u') {
                        // \uXXXX -> codepoint -> UTF-8
                        int cp = std::stoi(json.substr(k+2, 4), nullptr, 16);
                        clave += codepoint_a_utf8(cp);
                        k += 6;
                    } else if (esc == 'n') { clave += '\n'; k += 2; }
                    else if (esc == 't') { clave += '\t'; k += 2; }
                    else if (esc == 'r') { clave += '\r'; k += 2; }
                    else if (esc == 'b') { clave += '\b'; k += 2; }
                    else if (esc == 'f') { clave += '\f'; k += 2; }
                    else if (esc == '/') { clave += '/'; k += 2; }
                    else { clave += esc; k += 2; }  // \" \\ etc.
                } else {
                    clave += json[k];
                    k++;
                }
            }
            // buscar los dos puntos y el numero
            size_t dp = json.find(':', k);
            size_t ini = json.find_first_of("0123456789", dp);
            size_t fin = json.find_first_not_of("0123456789", ini);
            int id = std::stoi(json.substr(ini, fin - ini));

            id_a_token[id] = clave;
            i = fin;
        }
    }

    // Codepoint -> secuencia de bytes UTF-8
    std::string codepoint_a_utf8(int cp) {
        std::string r;
        if (cp < 0x80) r += (char)cp;
        else if (cp < 0x800) {
            r += (char)(0xC0 | (cp >> 6));
            r += (char)(0x80 | (cp & 0x3F));
        } else {
            r += (char)(0xE0 | (cp >> 12));
            r += (char)(0x80 | ((cp >> 6) & 0x3F));
            r += (char)(0x80 | (cp & 0x3F));
        }
        return r;
    }
};

// ----------------------------------------------------------------------
// SAMPLING: elegir el siguiente token a partir de la ultima fila de logits
// ----------------------------------------------------------------------

// GREEDY: el token de mayor logit. Determinista.
int muestrear_greedy(const Matrix& logits) {
    int fila = logits.rows - 1;
    int mejor = 0;
    float mejor_val = logits.at(fila, 0);
    for (int j = 1; j < logits.cols; j++)
        if (logits.at(fila, j) > mejor_val) {
            mejor_val = logits.at(fila, j);
            mejor = j;
        }
    return mejor;
}

// TEMPERATURA + TOP-K: mas natural, con algo de aleatoriedad controlada.
//   temperatura: <1 conservador, >1 creativo. 0.8 es buen punto.
//   top_k: solo considera los k tokens mas probables (0 = todos).
int muestrear_temp_topk(const Matrix& logits, float temperatura, int top_k,
                        std::mt19937& rng) {
    int fila = logits.rows - 1;
    int V = logits.cols;

    // pares (logit, id)
    std::vector<std::pair<float,int>> v;
    v.reserve(V);
    for (int j = 0; j < V; j++)
        v.push_back({logits.at(fila, j) / temperatura, j});

    // top-k: quedarse con los k mayores
    if (top_k > 0 && top_k < V) {
        std::partial_sort(v.begin(), v.begin() + top_k, v.end(),
                          [](auto&a, auto&b){ return a.first > b.first; });
        v.resize(top_k);
    }

    // softmax sobre los candidatos (con el truco del maximo)
    float m = v[0].first;
    for (auto& p : v) m = std::max(m, p.first);
    float suma = 0;
    for (auto& p : v) { p.first = std::exp(p.first - m); suma += p.first; }
    for (auto& p : v) p.first /= suma;

    // muestrear segun las probabilidades
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    float r = dist(rng);
    float acum = 0;
    for (auto& p : v) {
        acum += p.first;
        if (r <= acum) return p.second;
    }
    return v.back().second;  // fallback
}

#endif // GENERACION_H
