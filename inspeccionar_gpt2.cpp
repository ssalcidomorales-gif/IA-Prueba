// inspeccionar_gpt2.cpp
// Abre el model.safetensors REAL de GPT-2 y lista sus tensores con sus formas.
// Sirve para confirmar que el lector funciona con el archivo de 548 MB,
// y para que veas con tus propios ojos la estructura del modelo.
//
// Compilar: g++ -std=c++17 -O2 inspeccionar_gpt2.cpp -o inspeccionar_gpt2
// Correr:   ./inspeccionar_gpt2  C:\Jarvis\model.safetensors
//   (o pon la ruta de tu archivo como argumento)

#include "safetensors.h"
#include <iostream>
#include <iomanip>

int main(int argc, char** argv) {
    // Ruta del modelo: por argumento, o un valor por defecto
    std::string ruta = "C:\\Jarvis\\model.safetensors";
    if (argc >= 2) ruta = argv[1];

    std::cout << "Abriendo: " << ruta << "\n\n";

    try {
        SafeTensors st(ruta);

        auto nombres = st.nombres();
        std::cout << "Total de tensores: " << nombres.size() << "\n\n";

        // Mostrar cada tensor con su forma y dtype
        std::cout << std::left << std::setw(40) << "NOMBRE"
                  << std::setw(10) << "DTYPE" << "FORMA\n";
        std::cout << std::string(70, '-') << "\n";

        uint64_t total_params = 0;
        for (const auto& nombre : nombres) {
            const InfoTensor& info = st.info(nombre);

            std::cout << std::left << std::setw(40) << nombre
                      << std::setw(10) << info.dtype << "[";
            uint64_t params = 1;
            for (size_t i = 0; i < info.shape.size(); i++) {
                std::cout << info.shape[i];
                if (i + 1 < info.shape.size()) std::cout << ", ";
                params *= info.shape[i];
            }
            std::cout << "]\n";
            total_params += params;
        }

        std::cout << std::string(70, '-') << "\n";
        std::cout << "Total de parametros: " << total_params
                  << "  (~" << (total_params / 1000000) << "M)\n\n";

        // Verificacion concreta: el token embedding "wte.weight"
        // debe ser [50257 x 768]. Es la tabla que convierte cada token
        // en su vector de 768 numeros.
        std::cout << "Verificando el token embedding (wte.weight):\n";
        Matrix wte = st.getTensor("wte.weight");
        std::cout << "  Forma: [" << wte.rows << " x " << wte.cols << "]";
        if (wte.rows == 50257 && wte.cols == 768)
            std::cout << "  <- correcto!\n";
        else
            std::cout << "  <- inesperado (revisa)\n";

        std::cout << "  Primeros 5 valores de la fila 0: ";
        for (int i = 0; i < 5; i++)
            std::cout << std::fixed << std::setprecision(4)
                      << wte.at(0, i) << " ";
        std::cout << "\n\n";

        std::cout << "Si ves esto sin errores, el lector funciona con el "
                     "modelo real. Fin de semana 2 completado.\n";

    } catch (const std::exception& e) {
        std::cout << "ERROR: " << e.what() << "\n";
        std::cout << "\nRevisa que la ruta del archivo sea correcta y que "
                     "la descarga haya terminado (debe pesar ~548 MB).\n";
        return 1;
    }

    return 0;
}
