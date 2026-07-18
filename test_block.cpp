// test_block.cpp
// Verifica GELU, feedforward y el bloque transformer completo.
// Compilar: g++ -std=c++17 -O2 test_block.cpp -o test_block

#include "block.h"
#include <iostream>
#include <cmath>

int pruebas_ok = 0, pruebas_totales = 0;

bool casi_igual(float a, float b, float tol = 1e-3f) {
    return std::fabs(a - b) < tol;
}

void afirmar(const std::string& nombre, bool cond) {
    pruebas_totales++;
    if (cond) { std::cout << "[OK]    " << nombre << "\n"; pruebas_ok++; }
    else      { std::cout << "[FALLO] " << nombre << "\n"; }
}

int main() {
    std::cout << "=== Tests del bloque transformer ===\n\n";

    // --- Test 1: GELU(0) = 0 ---
    {
        Matrix x = {{0}};
        Matrix r = gelu(x);
        afirmar("GELU(0) = 0", casi_igual(r.at(0,0), 0.0f));
    }

    // --- Test 2: valores conocidos de GELU ---
    // GELU(1) ~= 0.8412, GELU(-1) ~= -0.1588
    // (con la aproximacion tanh de GPT-2)
    {
        Matrix x = {{1, -1}};
        Matrix r = gelu(x);
        afirmar("GELU(1) ~= 0.8412", casi_igual(r.at(0,0), 0.8412f));
        afirmar("GELU(-1) ~= -0.1588", casi_igual(r.at(0,1), -0.1588f));
    }

    // --- Test 3: GELU es (casi) identidad para positivos grandes ---
    // GELU(5) deberia ser muy cercano a 5.
    {
        Matrix x = {{5}};
        Matrix r = gelu(x);
        afirmar("GELU(5) ~= 5 (casi identidad)", casi_igual(r.at(0,0), 5.0f, 0.01f));
    }

    // --- Test 4: GELU aplasta negativos grandes hacia 0 ---
    // GELU(-5) deberia ser muy cercano a 0.
    {
        Matrix x = {{-5}};
        Matrix r = gelu(x);
        afirmar("GELU(-5) ~= 0 (aplasta negativos)", casi_igual(r.at(0,0), 0.0f, 0.01f));
    }

    // --- Test 5: feedforward preserva la forma [n x d] ---
    {
        int n = 3, d = 4, d_ff = 8;
        Matrix entrada(n, d);
        for (int i = 0; i < n*d; i++) entrada.data[i] = 0.1f * i;
        Matrix c_fc_w(d, d_ff);   for (int i=0;i<d*d_ff;i++) c_fc_w.data[i]=0.01f*i;
        Matrix c_fc_b(1, d_ff);
        Matrix c_proj_w(d_ff, d); for (int i=0;i<d_ff*d;i++) c_proj_w.data[i]=0.01f*i;
        Matrix c_proj_b(1, d);
        Matrix r = feedforward(entrada, c_fc_w, c_fc_b, c_proj_w, c_proj_b);
        afirmar("feedforward preserva forma [3 x 4]", r.rows==3 && r.cols==4);
    }

    // --- Test 6: el bloque transformer preserva la forma [n x d] ---
    // Y con residuales, la salida NO debe ser identica a la entrada
    // (algo tiene que haber cambiado al pasar por atencion y feedforward).
    {
        int n = 2, d = 4, cabezas = 2;
        Matrix x(n, d);
        for (int i = 0; i < n*d; i++) x.data[i] = 0.1f * (i+1);

        PesosBloque p;
        // layernorms: gamma=1, beta=0 (no distorsionan)
        p.ln_1_w = Matrix(1,d); for(int i=0;i<d;i++) p.ln_1_w.data[i]=1;
        p.ln_1_b = Matrix(1,d);
        p.ln_2_w = Matrix(1,d); for(int i=0;i<d;i++) p.ln_2_w.data[i]=1;
        p.ln_2_b = Matrix(1,d);
        // atencion
        p.c_attn_w = Matrix(d,3*d); for(int i=0;i<d*3*d;i++) p.c_attn_w.data[i]=0.01f*i;
        p.c_attn_b = Matrix(1,3*d);
        p.c_proj_w = Matrix(d,d);   for(int i=0;i<d*d;i++)   p.c_proj_w.data[i]=0.01f*i;
        p.c_proj_b = Matrix(1,d);
        // feedforward
        p.c_fc_w = Matrix(d,4*d);   for(int i=0;i<d*4*d;i++) p.c_fc_w.data[i]=0.01f*i;
        p.c_fc_b = Matrix(1,4*d);
        p.mlp_proj_w = Matrix(4*d,d); for(int i=0;i<4*d*d;i++) p.mlp_proj_w.data[i]=0.01f*i;
        p.mlp_proj_b = Matrix(1,d);

        Matrix salida = bloque_transformer(x, p, cabezas);
        afirmar("bloque preserva forma [2 x 4]", salida.rows==2 && salida.cols==4);

        bool cambio = false;
        for (int i = 0; i < n*d; i++)
            if (!casi_igual(salida.data[i], x.data[i])) cambio = true;
        afirmar("el bloque transforma la entrada (no es identidad)", cambio);
    }

    std::cout << "\n=== " << pruebas_ok << "/" << pruebas_totales
              << " pruebas pasaron ===\n";
    if (pruebas_ok == pruebas_totales)
        std::cout << "Bloque transformer completo. Solo falta apilar 12 y la salida.\n";

    return (pruebas_ok == pruebas_totales) ? 0 : 1;
}
