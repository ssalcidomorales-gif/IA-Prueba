// bpe.h
// Encoder BPE (Byte Pair Encoding): convierte texto a token IDs.
//
// Hasta ahora solo teniamos el DECODER (tokens -> texto). Este es el
// camino inverso, que permite escribir prompts normales en vez de meter
// IDs a mano.
//
// COMO FUNCIONA BPE
//
// 1. PRE-TOKENIZACION
//    El texto se parte en trozos con una expresion regular: palabras,
//    numeros, puntuacion, espacios. Cada trozo se tokeniza por separado,
//    lo que evita fusiones absurdas entre palabras distintas.
//
// 2. A BYTES DISFRAZADOS
//    Cada trozo se convierte a sus bytes UTF-8, y cada byte a su
//    caracter "disfrazado" (el espacio 0x20 -> 'G con punto', etc).
//    Es el mismo mapeo que ya usa el decoder, pero al reves.
//
// 3. FUSIONES
//    Se empieza con cada caracter como un simbolo suelto y se van
//    fusionando pares segun las reglas del archivo 'merges', SIEMPRE
//    en orden de prioridad. Si la regla numero 5 dice fusionar "Gt",
//    esa se aplica antes que la regla 100.
//
//    Ejemplo con "Ghello":
//      inicio:  G h e l l o
//      fusion:  G h  ->  Gh        (si esa regla tiene prioridad alta)
//      fusion:  l l  ->  ll
//      ...hasta que ninguna regla aplique
//
// 4. A IDs
//    Cada simbolo final se busca en el vocabulario.

#ifndef BPE_H
#define BPE_H

#include <string>
#include <vector>
#include <map>
#include <unordered_map>
#include <algorithm>
#include <cctype>
#include <cstring>

class BPE {
public:
    // Construye el tokenizador a partir de los datos del GGUF.
    //   tokens: tokenizer.ggml.tokens (el vocabulario)
    //   merges: tokenizer.ggml.merges (las reglas, "simbolo1 simbolo2")
    void construir(const std::vector<std::string>& tokens,
                   const std::vector<std::string>& merges) {
        vocab_ = tokens;

        // Mapa inverso: texto del token -> ID
        texto_a_id_.clear();
        texto_a_id_.reserve(tokens.size() * 2);
        for (size_t i = 0; i < tokens.size(); i++)
            texto_a_id_[tokens[i]] = (int)i;

        // Las reglas de fusion, con su prioridad (el indice en el archivo)
        rango_.clear();
        rango_.reserve(merges.size() * 2);
        for (size_t i = 0; i < merges.size(); i++) {
            // Cada linea es "izquierda derecha" separados por un espacio
            size_t esp = merges[i].find(' ');
            if (esp == std::string::npos) continue;
            std::string izq = merges[i].substr(0, esp);
            std::string der = merges[i].substr(esp + 1);
            rango_[izq + '\x01' + der] = (int)i;   // separador que no aparece en el texto
        }

        construir_byte_encoder();
    }

    bool listo() const { return !vocab_.empty() && !rango_.empty(); }
    size_t tam_vocab() const { return vocab_.size(); }

    // ------------------------------------------------------------------
    // TEXTO -> TOKEN IDS
    // ------------------------------------------------------------------
    std::vector<int> codificar(const std::string& texto) {
        std::vector<int> ids;

        // 1. Pre-tokenizar en trozos
        std::vector<std::string> trozos = pre_tokenizar(texto);

        for (const std::string& trozo : trozos) {
            // 2. Convertir a caracteres disfrazados
            std::string disfrazado = disfrazar_bytes(trozo);

            // 3. Aplicar las fusiones BPE
            std::vector<std::string> simbolos = aplicar_bpe(disfrazado);

            // 4. Buscar cada simbolo en el vocabulario
            for (const std::string& s : simbolos) {
                auto it = texto_a_id_.find(s);
                if (it != texto_a_id_.end()) {
                    ids.push_back(it->second);
                } else {
                    // Si no esta (raro), meter byte por byte
                    for (unsigned char c : s) {
                        std::string uno = disfrazar_bytes(std::string(1, (char)c));
                        auto it2 = texto_a_id_.find(uno);
                        if (it2 != texto_a_id_.end()) ids.push_back(it2->second);
                    }
                }
            }
        }
        return ids;
    }

    // Busca el ID de un token especial por su texto exacto
    // (por ejemplo "<|im_start|>")
    int id_de(const std::string& token_texto) const {
        auto it = texto_a_id_.find(token_texto);
        return (it == texto_a_id_.end()) ? -1 : it->second;
    }

private:
    std::vector<std::string> vocab_;
    std::unordered_map<std::string, int> texto_a_id_;
    std::unordered_map<std::string, int> rango_;   // par fusionable -> prioridad
    std::map<int, std::string> byte_a_char_;       // byte -> caracter disfrazado

    // ------------------------------------------------------------------
    // El mapeo byte-level de GPT-2/Qwen (el mismo del decoder, al reves)
    // ------------------------------------------------------------------
    void construir_byte_encoder() {
        std::vector<int> bs, cs;
        for (int i = (int)'!'; i <= (int)'~'; i++) bs.push_back(i);
        for (int i = 0xA1; i <= 0xAC; i++) bs.push_back(i);
        for (int i = 0xAE; i <= 0xFF; i++) bs.push_back(i);
        cs = bs;
        int n = 0;
        for (int b = 0; b < 256; b++) {
            if (std::find(bs.begin(), bs.end(), b) == bs.end()) {
                bs.push_back(b);
                cs.push_back(256 + n);
                n++;
            }
        }
        byte_a_char_.clear();
        for (size_t i = 0; i < bs.size(); i++)
            byte_a_char_[bs[i]] = codepoint_a_utf8(cs[i]);
    }

    static std::string codepoint_a_utf8(int cp) {
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

    // Convierte los bytes crudos a sus caracteres disfrazados
    std::string disfrazar_bytes(const std::string& s) {
        std::string r;
        for (unsigned char c : s) {
            auto it = byte_a_char_.find((int)c);
            if (it != byte_a_char_.end()) r += it->second;
        }
        return r;
    }

    // ------------------------------------------------------------------
    // PRE-TOKENIZACION
    // ------------------------------------------------------------------
    // La regex original de GPT-2 es:
    //   's|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+
    //
    // La implementamos a mano en vez de con std::regex porque necesitamos
    // manejar UTF-8 y las clases \p{L} (letras) y \p{N} (numeros) de
    // Unicode, que std::regex no soporta bien.
    //
    // Las reglas, en orden:
    //   1. Contracciones del ingles ('s, 't, 're, 've, 'm, 'll, 'd)
    //   2. Un espacio opcional seguido de letras
    //   3. Un espacio opcional seguido de digitos
    //   4. Un espacio opcional seguido de otros simbolos
    //   5. Espacios en bloque (dejando el ultimo para la palabra siguiente)
    std::vector<std::string> pre_tokenizar(const std::string& texto) {
        std::vector<std::string> trozos;
        size_t i = 0;
        size_t n = texto.size();

        while (i < n) {
            // --- Contracciones ---
            if (texto[i] == '\'' && i + 1 < n) {
                static const char* contr[] = {"'s","'t","'re","'ve","'m","'ll","'d"};
                bool encontrada = false;
                for (const char* c : contr) {
                    size_t len = std::strlen(c);
                    if (texto.compare(i, len, c) == 0) {
                        trozos.push_back(std::string(c));
                        i += len;
                        encontrada = true;
                        break;
                    }
                }
                if (encontrada) continue;
            }

            // --- Espacio opcional al inicio del trozo ---
            size_t inicio = i;
            bool con_espacio = false;
            if (texto[i] == ' ' && i + 1 < n && !es_espacio(texto[i+1])) {
                con_espacio = true;
                i++;
            }

            if (i >= n) {
                // Solo quedaba el espacio
                trozos.push_back(texto.substr(inicio));
                break;
            }

            // --- Letras ---
            if (es_letra_inicio(texto, i)) {
                while (i < n && es_letra_inicio(texto, i)) i += largo_utf8(texto[i]);
                trozos.push_back(texto.substr(inicio, i - inicio));
                continue;
            }

            // --- Digitos ---
            if (std::isdigit((unsigned char)texto[i])) {
                while (i < n && std::isdigit((unsigned char)texto[i])) i++;
                trozos.push_back(texto.substr(inicio, i - inicio));
                continue;
            }

            // --- Espacios en bloque ---
            if (es_espacio(texto[i]) && !con_espacio) {
                size_t ini_esp = i;
                while (i < n && es_espacio(texto[i])) i++;
                // Si despues de los espacios viene algo, el ultimo espacio
                // pertenece a la palabra siguiente
                if (i < n && (i - ini_esp) > 1) i--;
                trozos.push_back(texto.substr(ini_esp, i - ini_esp));
                continue;
            }

            // --- Otros simbolos (puntuacion, etc) ---
            while (i < n && !es_espacio(texto[i]) &&
                   !std::isdigit((unsigned char)texto[i]) &&
                   !es_letra_inicio(texto, i)) {
                i += largo_utf8(texto[i]);
            }
            if (i == inicio) i++;   // seguro contra bucle infinito
            trozos.push_back(texto.substr(inicio, i - inicio));
        }

        return trozos;
    }

    static bool es_espacio(char c) {
        return c==' ' || c=='\t' || c=='\n' || c=='\r' || c=='\v' || c=='\f';
    }

    static int largo_utf8(char c) {
        unsigned char u = (unsigned char)c;
        if (u < 0x80) return 1;
        if (u < 0xE0) return 2;
        if (u < 0xF0) return 3;
        return 4;
    }

    // Aproximacion de \p{L}: letras ASCII, guion bajo, y cualquier
    // caracter multibyte (asumimos que los no-ASCII son letras, que es
    // cierto para la mayoria de los idiomas)
    static bool es_letra_inicio(const std::string& s, size_t i) {
        unsigned char c = (unsigned char)s[i];
        if (c >= 0x80) return true;                 // multibyte: tratarlo como letra
        return std::isalpha(c) != 0;
    }

    // ------------------------------------------------------------------
    // EL ALGORITMO BPE
    // ------------------------------------------------------------------
    // Empieza con cada caracter como simbolo suelto. Repetidamente busca
    // el par adyacente con MENOR rango (mayor prioridad en el archivo de
    // merges) y lo fusiona. Para cuando ningun par es fusionable.
    std::vector<std::string> aplicar_bpe(const std::string& palabra) {
        // Partir en caracteres UTF-8
        std::vector<std::string> simbolos;
        size_t i = 0;
        while (i < palabra.size()) {
            int len = largo_utf8(palabra[i]);
            simbolos.push_back(palabra.substr(i, len));
            i += len;
        }
        if (simbolos.size() < 2) return simbolos;

        while (true) {
            // Buscar el par con mejor prioridad (rango mas bajo)
            int mejor_rango = -1;
            size_t mejor_pos = 0;

            for (size_t k = 0; k + 1 < simbolos.size(); k++) {
                std::string clave = simbolos[k] + '\x01' + simbolos[k+1];
                auto it = rango_.find(clave);
                if (it != rango_.end()) {
                    if (mejor_rango < 0 || it->second < mejor_rango) {
                        mejor_rango = it->second;
                        mejor_pos = k;
                    }
                }
            }

            if (mejor_rango < 0) break;   // ya no hay nada que fusionar

            // Fusionar ese par
            simbolos[mejor_pos] = simbolos[mejor_pos] + simbolos[mejor_pos+1];
            simbolos.erase(simbolos.begin() + mejor_pos + 1);

            if (simbolos.size() == 1) break;
        }

        return simbolos;
    }
};

#endif // BPE_H
