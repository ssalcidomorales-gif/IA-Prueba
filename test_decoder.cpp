#include "generacion.h"
#include <iostream>

int main() {
    // Verificar el byte_decoder: 'Ġ' (U+0120 = 288) debe mapear a byte 32 (espacio)
    auto bd = construir_byte_decoder();
    
    std::cout << "=== Test del byte decoder ===\n";
    // U+0120 es 'Ġ', codepoint 288
    std::cout << "codepoint 288 (Gdot) -> byte " << bd[288] 
              << " (esperado 32 = espacio)\n";
    // 'A' = 65 debe mapear a si mismo
    std::cout << "codepoint 65 (A) -> byte " << bd[65] 
              << " (esperado 65)\n";
    // U+010A es 'Ċ', codepoint 266 -> byte 10 (newline)
    std::cout << "codepoint 266 (Cdot) -> byte " << bd[266]
              << " (esperado 10 = newline)\n";
    
    bool ok = (bd[288]==32) && (bd[65]==65) && (bd[266]==10);
    
    // Test de decodificacion completa con un vocab simulado
    std::cout << "\n=== Test de decodificacion ===\n";
    Tokenizer tok;
    tok.byte_decoder = bd;
    // Simular: token 0 = "Hello", token 1 = "Ġworld" (con Ġ = espacio)
    // 'Ġ' en UTF-8 es 0xC4 0xA0
    tok.id_a_token[0] = "Hello";
    tok.id_a_token[1] = "\xC4\xA0world";  // Ġworld
    
    std::string r = tok.decodificar({0, 1});
    std::cout << "decodificar {0,1} = '" << r << "' (esperado 'Hello world')\n";
    ok = ok && (r == "Hello world");
    
    std::cout << "\n" << (ok ? "TODO CORRECTO" : "HAY FALLOS") << "\n";
    return ok ? 0 : 1;
}
