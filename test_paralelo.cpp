// test_paralelo.cpp
// Verifica que el matmul paralelo da resultados identicos al secuencial,
// y mide cuanto mas rapido es. No necesita el modelo.

#include "paralelo.h"
#include <iostream>
#include <iomanip>
#include <chrono>
#include <cmath>

bool casi(float a, float b, float tol = 1e-3f) { return std::fabs(a-b) < tol; }

int main() {
    std::cout << "=== Test del matmul paralelo ===\n\n";
    std::cout << "Nucleos detectados: " << Paralelo::num_hilos() << "\n\n";

    int fallos = 0;

    // --- Test 1: correccion con matriz chica (calculada a mano) ---
    {
        Matrix A = {{1, 2}, {3, 4}};
        Matrix B = {{5, 6}, {7, 8}};
        Matrix r = matmul_paralelo(A, B);
        bool ok = casi(r.at(0,0),19) && casi(r.at(0,1),22) &&
                  casi(r.at(1,0),43) && casi(r.at(1,1),50);
        std::cout << (ok ? "[OK]    " : "[FALLO] ")
                  << "matmul 2x2 correcto\n";
        if (!ok) fallos++;
    }

    // --- Test 2: paralelo == secuencial con matriz grande ---
    {
        int m = 400, n = 300, p = 500;
        Matrix A(m, n), B(n, p);
        for (int i = 0; i < m*n; i++) A.data[i] = 0.001f * (i % 97) - 0.05f;
        for (int i = 0; i < n*p; i++) B.data[i] = 0.002f * (i % 53) - 0.03f;

        Matrix seq = A.matmul(B);
        Matrix par = matmul_paralelo(A, B);

        float max_dif = 0;
        for (int i = 0; i < m*p; i++)
            max_dif = std::max(max_dif, std::fabs(seq.data[i] - par.data[i]));

        bool ok = (max_dif < 1e-3f);
        std::cout << (ok ? "[OK]    " : "[FALLO] ")
                  << "paralelo == secuencial (dif max: " << max_dif << ")\n";
        if (!ok) fallos++;
    }

    // --- Test 3: matmul_transpuesta == matmul con transpose ---
    {
        int m = 3, n = 200, p = 400;
        Matrix A(m, n), B(p, n);
        for (int i = 0; i < m*n; i++) A.data[i] = 0.003f * (i % 71) - 0.1f;
        for (int i = 0; i < p*n; i++) B.data[i] = 0.001f * (i % 89) - 0.04f;

        Matrix esperado = A.matmul(B.transpose());
        Matrix obtenido = matmul_transpuesta_paralelo(A, B);

        float max_dif = 0;
        for (int i = 0; i < m*p; i++)
            max_dif = std::max(max_dif, std::fabs(esperado.data[i] - obtenido.data[i]));

        bool ok = (max_dif < 1e-3f);
        std::cout << (ok ? "[OK]    " : "[FALLO] ")
                  << "matmul_transpuesta correcto (dif max: " << max_dif << ")\n";
        if (!ok) fallos++;
    }

    // --- Test 4: caso real - proyeccion a logits (1 x 768) * (50257 x 768)^T ---
    {
        Matrix x(1, 768);
        Matrix wte(50257, 768);
        for (int i = 0; i < 768; i++) x.data[i] = 0.01f * (i % 31) - 0.1f;
        for (int i = 0; i < 50257*768; i++) wte.data[i] = 0.0001f * (i % 127) - 0.005f;

        auto t1 = std::chrono::high_resolution_clock::now();
        Matrix seq = x.matmul(wte.transpose());
        auto t2 = std::chrono::high_resolution_clock::now();
        Matrix par = matmul_transpuesta_paralelo(x, wte);
        auto t3 = std::chrono::high_resolution_clock::now();

        double ms_seq = std::chrono::duration<double, std::milli>(t2-t1).count();
        double ms_par = std::chrono::duration<double, std::milli>(t3-t2).count();

        float max_dif = 0;
        for (int i = 0; i < 50257; i++)
            max_dif = std::max(max_dif, std::fabs(seq.data[i] - par.data[i]));

        bool ok = (max_dif < 1e-2f);
        std::cout << (ok ? "[OK]    " : "[FALLO] ")
                  << "proyeccion a logits correcta (dif max: " << max_dif << ")\n\n";
        if (!ok) fallos++;

        std::cout << "--- Proyeccion a logits (la operacion mas pesada) ---\n";
        std::cout << std::fixed << std::setprecision(1);
        std::cout << "  Con transpose + secuencial: " << ms_seq << " ms\n";
        std::cout << "  Transpuesta implicita + paralelo: " << ms_par << " ms\n";
        if (ms_par > 0)
            std::cout << "  Mejora: " << std::setprecision(2)
                      << (ms_seq/ms_par) << "x\n";
    }

    // --- Benchmark: matmul tipico del feed-forward ---
    {
        std::cout << "\n--- Feed-forward tipico: [1 x 768] * [768 x 3072] ---\n";
        Matrix A(1, 768), B(768, 3072);
        for (int i = 0; i < 768; i++) A.data[i] = 0.01f * (i % 31);
        for (int i = 0; i < 768*3072; i++) B.data[i] = 0.0001f * (i % 97);

        int repes = 50;
        auto t1 = std::chrono::high_resolution_clock::now();
        for (int r = 0; r < repes; r++) { volatile auto x = A.matmul(B); }
        auto t2 = std::chrono::high_resolution_clock::now();
        for (int r = 0; r < repes; r++) { volatile auto x = matmul_paralelo(A, B); }
        auto t3 = std::chrono::high_resolution_clock::now();

        double ms_seq = std::chrono::duration<double, std::milli>(t2-t1).count()/repes;
        double ms_par = std::chrono::duration<double, std::milli>(t3-t2).count()/repes;
        std::cout << std::setprecision(2);
        std::cout << "  Secuencial: " << ms_seq << " ms\n";
        std::cout << "  Paralelo:   " << ms_par << " ms\n";
        if (ms_par > 0) std::cout << "  Mejora: " << (ms_seq/ms_par) << "x\n";
    }

    std::cout << "\n=== " << (fallos == 0 ? "TODO CORRECTO" : "HAY FALLOS") << " ===\n";
    return fallos;
}
