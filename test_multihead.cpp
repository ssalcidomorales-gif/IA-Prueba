// test_multihead.cpp
// Verifica la capa lineal Conv1D y el multi-head attention.
// Compilar: g++ -std=c++17 -O2 test_multihead.cpp -o test_multihead

#include "attention.h"
#include <iostream>
#include <cmath>

int pruebas_ok = 0, pruebas_totales = 0;

bool casi_igual(float a, float b, float tol = 1e-4f) {
    return std::fabs(a - b) < tol;
}

void afirmar(const std::string& nombre, bool cond) {
    pruebas_totales++;
    if (cond) { std::cout << "[OK]    " << nombre << "\n"; pruebas_ok++; }
    else      { std::cout << "[FALLO] " << nombre << "\n"; }
}

int main() {
    std::cout << "=== Tests de multi-head attention ===\n\n";

    // --- Test 1: capa lineal Conv1D = entrada * W + bias ---
    // entrada [1x2] = [1, 2]
    // W [2x3] = [[1, 0, 1], [0, 1, 1]]
    // entrada*W = [1*1+2*0, 1*0+2*1, 1*1+2*1] = [1, 2, 3]
    // + bias [10, 20, 30] = [11, 22, 33]
    {
        Matrix entrada = {{1, 2}};
        Matrix W = {{1, 0, 1}, {0, 1, 1}};
        Matrix bias = {{10, 20, 30}};
        Matrix r = lineal_conv1d(entrada, W, bias);
        afirmar("conv1d = entrada*W + bias",
                casi_igual(r.at(0,0), 11) &&
                casi_igual(r.at(0,1), 22) &&
                casi_igual(r.at(0,2), 33));
    }

    // --- Test 2: sub_columnas extrae el bloque correcto ---
    {
        Matrix m = {{1, 2, 3, 4}, {5, 6, 7, 8}};
        Matrix bloque = sub_columnas(m, 1, 2);  // columnas 1 y 2
        afirmar("sub_columnas extrae [2,3] y [6,7]",
                bloque.rows == 2 && bloque.cols == 2 &&
                bloque.at(0,0)==2 && bloque.at(0,1)==3 &&
                bloque.at(1,0)==6 && bloque.at(1,1)==7);
    }

    // --- Test 3: multi-head con 1 cabeza == una cabeza suelta ---
    // Esta es la prueba clave. Si armamos multi-head con una sola cabeza
    // y pesos identidad (para que la proyeccion no cambie nada), el
    // resultado debe coincidir con llamar atencion_una_cabeza directamente.
    {
        int n = 3, d = 2;

        // entrada [3x2]
        Matrix entrada = {{1, 0}, {0, 1}, {1, 1}};

        // c_attn: queremos que Q=K=V=entrada. Con 1 cabeza, qkv debe ser
        // [entrada | entrada | entrada] = [3 x 6]. Eso se logra con una W
        // [2 x 6] que copia la entrada tres veces, y bias 0.
        Matrix c_attn_w = {{1, 0, 1, 0, 1, 0},
                           {0, 1, 0, 1, 0, 1}};
        Matrix c_attn_b(1, 6);  // ceros

        // c_proj identidad [2x2], bias 0 -> no cambia la salida
        Matrix c_proj_w = {{1, 0}, {0, 1}};
        Matrix c_proj_b(1, 2);  // ceros

        Matrix salida_mh = multi_head_attention(
            entrada, c_attn_w, c_attn_b, c_proj_w, c_proj_b, 1);

        // referencia: una cabeza con Q=K=V=entrada
        Matrix salida_ref = atencion_una_cabeza(entrada, entrada, entrada);

        bool coincide = true;
        for (int i = 0; i < n; i++)
            for (int j = 0; j < d; j++)
                if (!casi_igual(salida_mh.at(i,j), salida_ref.at(i,j)))
                    coincide = false;
        afirmar("multi-head(1 cabeza) == una cabeza suelta", coincide);
    }

    // --- Test 4: forma de salida correcta con varias cabezas ---
    // entrada [4 x 8], 2 cabezas (dim_cabeza = 4).
    // Pesos aleatorios simples; solo verificamos la FORMA.
    {
        int n = 4, d = 8;
        Matrix entrada(n, d);
        for (int i = 0; i < n*d; i++) entrada.data[i] = 0.01f * i;

        Matrix c_attn_w(d, 3*d);
        for (int i = 0; i < d*3*d; i++) c_attn_w.data[i] = 0.001f * i;
        Matrix c_attn_b(1, 3*d);

        Matrix c_proj_w(d, d);
        for (int i = 0; i < d*d; i++) c_proj_w.data[i] = 0.001f * i;
        Matrix c_proj_b(1, d);

        Matrix salida = multi_head_attention(
            entrada, c_attn_w, c_attn_b, c_proj_w, c_proj_b, 2);
        afirmar("salida multi-head tiene forma [4 x 8]",
                salida.rows == 4 && salida.cols == 8);
    }

    std::cout << "\n=== " << pruebas_ok << "/" << pruebas_totales
              << " pruebas pasaron ===\n";
    if (pruebas_ok == pruebas_totales)
        std::cout << "Multi-head attention listo. El bloque de atencion completo.\n";

    return (pruebas_ok == pruebas_totales) ? 0 : 1;
}
