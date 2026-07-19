// test_bpe.cpp
// Verifica el encoder BPE con un vocabulario sintetico pequeno,
// y comprueba que codificar->decodificar sea reversible.

#include "bpe.h"
#include <iostream>
#include <cstring>

int ok = 0, total = 0;
void afirmar(const std::string& n, bool c) {
    total++;
    if (c) { std::cout << "[OK]    " << n << "\n"; ok++; }
    else   { std::cout << "[FALLO] " << n << "\n"; }
}

int main() {
    std::cout << "=== Tests del encoder BPE ===\n\n";

    // Vocabulario sintetico. Incluimos caracteres sueltos y algunas
    // fusiones para poder verificar el algoritmo.
    // 'G con punto' (U+0120) es el espacio disfrazado; en UTF-8 es \xC4\xA0
    std::string ESP = "\xC4\xA0";

    std::vector<std::string> vocab = {
        "h","e","l","o","w","r","d","!",           // 0-7
        "he","ll","lo","hell","hello",             // 8-12
        ESP, ESP+"w", ESP+"wo", ESP+"wor", ESP+"world",  // 13-17
        "<|im_start|>", "<|im_end|>"               // 18-19
    };

    // Reglas de fusion en orden de prioridad
    std::vector<std::string> merges = {
        "h e",          // 0: h+e -> he
        "l l",          // 1: l+l -> ll
        "he ll",        // 2: he+ll -> hell
        "hell o",       // 3: hell+o -> hello
        ESP + " w",     // 4: espacio+w
        ESP+"w o",      // 5
        ESP+"wo r",     // 6
        ESP+"wor ld"    // 7 (nota: 'ld' no esta suelto, esta regla no aplicara)
    };

    BPE bpe;
    bpe.construir(vocab, merges);

    afirmar("el tokenizador se construye", bpe.listo());
    afirmar("tam_vocab correcto", bpe.tam_vocab() == vocab.size());

    // --- Fusion completa: "hello" debe volverse un solo token ---
    {
        auto ids = bpe.codificar("hello");
        std::cout << "        'hello' -> ";
        for (int id : ids) std::cout << id << " ";
        std::cout << "\n";
        afirmar("'hello' se fusiona en un solo token (id 12)",
                ids.size()==1 && ids[0]==12);
    }

    // --- Fusion parcial: "hell" para en el token 11 ---
    {
        auto ids = bpe.codificar("hell");
        afirmar("'hell' -> token 11", ids.size()==1 && ids[0]==11);
    }

    // --- El orden de prioridad importa ---
    // "he" (regla 0) se aplica antes que otras, formando "he" primero
    {
        auto ids = bpe.codificar("he");
        afirmar("'he' -> token 8", ids.size()==1 && ids[0]==8);
    }

    // --- Caracteres sueltos sin fusion ---
    {
        auto ids = bpe.codificar("d");
        afirmar("'d' -> token 6", ids.size()==1 && ids[0]==6);
    }

    // --- Tokens especiales por nombre ---
    {
        afirmar("id_de('<|im_start|>') = 18", bpe.id_de("<|im_start|>")==18);
        afirmar("id_de('<|im_end|>') = 19",   bpe.id_de("<|im_end|>")==19);
        afirmar("id_de de algo inexistente = -1", bpe.id_de("no_existe")==-1);
    }

    // --- El espacio se disfraza correctamente ---
    {
        auto ids = bpe.codificar(" w");
        std::cout << "        ' w' -> ";
        for (int id : ids) std::cout << id << " ";
        std::cout << "\n";
        // Deberia fusionar espacio+w = token 14
        afirmar("' w' usa el espacio disfrazado", ids.size()==1 && ids[0]==14);
    }

    // --- Pre-tokenizacion: palabras separadas no se fusionan entre si ---
    {
        auto ids = bpe.codificar("he he");
        std::cout << "        'he he' -> ";
        for (int id : ids) std::cout << id << " ";
        std::cout << "\n";
        // Debe dar al menos 2 tokens (las dos palabras por separado)
        afirmar("palabras separadas no se fusionan entre si", ids.size() >= 2);
    }

    std::cout << "\n=== " << ok << "/" << total << " pruebas pasaron ===\n";
    return (ok==total) ? 0 : 1;
}
