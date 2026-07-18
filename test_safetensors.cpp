// test_safetensors.cpp
// Crea un archivo .safetensors pequenito con valores que conocemos,
// lo lee con nuestra clase, y verifica que todo salga correcto.
// Asi probamos el lector sin necesitar el modelo real de 548 MB.
//
// Compilar: g++ -std=c++17 -O2 test_safetensors.cpp -o test_safetensors
// Correr:   ./test_safetensors

#include "safetensors.h"
#include <fstream>
#include <cstring>
#include <iostream>

// Escribe un archivo safetensors de prueba con dos tensores:
//   "peso"  -> [2 x 3] con valores 1..6
//   "sesgo" -> [3]     con valores 10, 20, 30
void crearArchivoDePrueba(const std::string& ruta) {
    // Los datos crudos, en el orden en que van en el bloque de datos
    std::vector<float> datos_peso  = {1, 2, 3, 4, 5, 6};      // 24 bytes
    std::vector<float> datos_sesgo = {10, 20, 30};            // 12 bytes

    // El header JSON. Los offsets son relativos al inicio del bloque de datos.
    // peso:  bytes [0, 24)
    // sesgo: bytes [24, 36)
    std::string json =
        "{"
        "\"peso\":{\"dtype\":\"F32\",\"shape\":[2,3],\"data_offsets\":[0,24]},"
        "\"sesgo\":{\"dtype\":\"F32\",\"shape\":[3],\"data_offsets\":[24,36]}"
        "}";

    // safetensors exige que el header este alineado; rellenamos con espacios
    // hasta un multiplo de 8 (opcional pero es lo que hace la libreria real).
    while (json.size() % 8 != 0) json += ' ';

    uint64_t tam_header = json.size();

    std::ofstream f(ruta, std::ios::binary);
    // 8 bytes: tamano del header
    f.write(reinterpret_cast<const char*>(&tam_header), 8);
    // el header
    f.write(json.data(), json.size());
    // los datos
    f.write(reinterpret_cast<const char*>(datos_peso.data()),
            datos_peso.size() * 4);
    f.write(reinterpret_cast<const char*>(datos_sesgo.data()),
            datos_sesgo.size() * 4);
    f.close();
}

int pruebas_ok = 0, pruebas_totales = 0;

void afirmar(const std::string& nombre, bool condicion) {
    pruebas_totales++;
    if (condicion) { std::cout << "[OK]    " << nombre << "\n"; pruebas_ok++; }
    else           { std::cout << "[FALLO] " << nombre << "\n"; }
}

int main() {
    std::cout << "=== Test del lector safetensors ===\n\n";

    const std::string ruta = "prueba.safetensors";
    crearArchivoDePrueba(ruta);

    SafeTensors st(ruta);

    // 1. Encontro los dos tensores
    afirmar("encontro 2 tensores", st.nombres().size() == 2);

    // 2. La forma del peso es [2 x 3]
    {
        const InfoTensor& info = st.info("peso");
        afirmar("forma de 'peso' es [2,3]",
                info.shape.size() == 2 && info.shape[0] == 2 && info.shape[1] == 3);
        afirmar("dtype de 'peso' es F32", info.dtype == "F32");
    }

    // 3. Los valores del peso son 1..6 en el orden correcto
    {
        Matrix peso = st.getTensor("peso");
        bool ok = peso.rows == 2 && peso.cols == 3 &&
                  peso.at(0,0) == 1 && peso.at(0,1) == 2 && peso.at(0,2) == 3 &&
                  peso.at(1,0) == 4 && peso.at(1,1) == 5 && peso.at(1,2) == 6;
        afirmar("valores de 'peso' correctos (1..6)", ok);
    }

    // 4. El sesgo (1D) se lee como [1 x 3] con 10,20,30
    {
        Matrix sesgo = st.getTensor("sesgo");
        bool ok = sesgo.rows == 1 && sesgo.cols == 3 &&
                  sesgo.at(0,0) == 10 && sesgo.at(0,1) == 20 && sesgo.at(0,2) == 30;
        afirmar("valores de 'sesgo' correctos (10,20,30)", ok);
    }

    // 5. Pedir un tensor inexistente lanza excepcion
    {
        bool lanzo = false;
        try { st.getTensor("no_existe"); }
        catch (const std::exception&) { lanzo = true; }
        afirmar("tensor inexistente lanza error", lanzo);
    }

    std::cout << "\n=== " << pruebas_ok << "/" << pruebas_totales
              << " pruebas pasaron ===\n";

    // Limpiar el archivo de prueba
    std::remove(ruta.c_str());

    return (pruebas_ok == pruebas_totales) ? 0 : 1;
}
