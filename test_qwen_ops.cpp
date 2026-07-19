// test_qwen_ops.cpp
// Verifica RMSNorm, RoPE y SwiGLU. El test estrella es el de RoPE:
// comprueba la propiedad que lo hace funcionar (invarianza traslacional).

#include "qwen_ops.h"
#include <iostream>
#include <iomanip>
#include <cmath>

int ok = 0, total = 0;
void afirmar(const std::string& n, bool c) {
    total++;
    if (c) { std::cout << "[OK]    " << n << "\n"; ok++; }
    else { std::cout << "[FALLO] " << n << "\n"; }
}
bool casi(float a, float b, float tol = 1e-4f) { return std::fabs(a - b) < tol; }

float producto_punto(const Matrix& a, int fa, const Matrix& b, int fb) {
    float s = 0;
    for (int j = 0; j < a.cols; j++) s += a.at(fa, j) * b.at(fb, j);
    return s;
}

int main() {
    std::cout << "=== Tests de las operaciones de Qwen ===\n\n";

    // ================= RMSNorm =================
    std::cout << "--- RMSNorm ---\n";

    // Fila [3, 4]: promedio de cuadrados = (9+16)/2 = 12.5
    // rms = sqrt(12.5) = 3.5355
    // salida = [3/3.5355, 4/3.5355] = [0.8485, 1.1314]
    {
        Matrix x = { {3, 4} };
        Matrix g = { {1, 1} };
        Matrix r = rmsnorm(x, g, 0.0f);
        afirmar("rmsnorm [3,4] con gamma=1",
            casi(r.at(0, 0), 0.84853f, 1e-3f) &&
            casi(r.at(0, 1), 1.13137f, 1e-3f));
    }

    // gamma escala la salida
    {
        Matrix x = { {3, 4} };
        Matrix g = { {2, 0.5f} };
        Matrix r = rmsnorm(x, g, 0.0f);
        afirmar("rmsnorm aplica gamma",
            casi(r.at(0, 0), 0.84853f * 2, 1e-3f) &&
            casi(r.at(0, 1), 1.13137f * 0.5f, 1e-3f));
    }

    // A diferencia de LayerNorm, RMSNorm NO centra en cero:
    // una fila con todos los valores iguales y positivos sigue positiva
    {
        Matrix x = { {5, 5, 5, 5} };
        Matrix g = { {1, 1, 1, 1} };
        Matrix r = rmsnorm(x, g, 0.0f);
        // rms = 5, asi que salida = [1,1,1,1]
        bool bien = true;
        for (int j = 0; j < 4; j++) if (!casi(r.at(0, j), 1.0f)) bien = false;
        afirmar("rmsnorm NO centra en cero (a diferencia de layernorm)", bien);
    }

    // Cada fila se normaliza independiente
    {
        Matrix x = { {3, 4}, {30, 40} };
        Matrix g = { {1, 1} };
        Matrix r = rmsnorm(x, g, 0.0f);
        // la segunda fila es 10x la primera, pero tras normalizar son iguales
        afirmar("rmsnorm normaliza cada fila por separado",
            casi(r.at(0, 0), r.at(1, 0), 1e-3f) &&
            casi(r.at(0, 1), r.at(1, 1), 1e-3f));
    }

    // ================= RoPE =================
    std::cout << "\n--- RoPE ---\n";

    // En la posicion 0, theta = 0, asi que cos=1 y sin=0: no rota nada
    {
        Matrix x = { {1, 2, 3, 4} };
        Matrix original = x;
        aplicar_rope(x, 0);
        bool bien = true;
        for (int j = 0; j < 4; j++)
            if (!casi(x.at(0, j), original.at(0, j))) bien = false;
        afirmar("RoPE en posicion 0 no cambia nada", bien);
    }

    // Rotacion preserva la norma de cada par.
    // OJO: con la convencion NeoX el par de la dimension 0 es (0, d/2),
    // NO (0,1). Con d=4, el par es (0,2).
    {
        Matrix x = { {3, 1, 4, 0} };   // par (0,2) = (3,4), par (1,3) = (1,0)
        float antes = std::sqrt(3 * 3 + 4 * 4);
        aplicar_rope(x, 7);
        float despues = std::sqrt(
            x.at(0, 0) * x.at(0, 0) + x.at(0, 2) * x.at(0, 2));
        afirmar("RoPE preserva la norma de cada par (NeoX: j con j+d/2)",
            casi(antes, despues, 1e-3f));
    }

    // EL TEST CLAVE: invarianza traslacional.
    // El producto punto entre Q en posicion m y K en posicion n debe
    // depender SOLO de (m - n), no de los valores absolutos.
    // Verificamos que (m=5,n=3) da lo mismo que (m=12,n=10): ambos con
    // distancia 2.
    {
        int d = 8;
        Matrix q_base(1, d), k_base(1, d);
        for (int j = 0; j < d; j++) {
            q_base.at(0, j) = 0.3f * (j + 1) - 0.5f;
            k_base.at(0, j) = 0.2f * (j + 1) + 0.1f;
        }

        // Caso A: posiciones 5 y 3 (distancia 2)
        Matrix qa = q_base, ka = k_base;
        aplicar_rope(qa, 5);
        aplicar_rope(ka, 3);
        float dot_a = producto_punto(qa, 0, ka, 0);

        // Caso B: posiciones 12 y 10 (misma distancia 2)
        Matrix qb = q_base, kb = k_base;
        aplicar_rope(qb, 12);
        aplicar_rope(kb, 10);
        float dot_b = producto_punto(qb, 0, kb, 0);

        // Caso C: distancia distinta (posiciones 5 y 1, distancia 4)
        Matrix qc = q_base, kc = k_base;
        aplicar_rope(qc, 5);
        aplicar_rope(kc, 1);
        float dot_c = producto_punto(qc, 0, kc, 0);

        std::cout << "        dot(pos 5,3)  = " << std::fixed
            << std::setprecision(6) << dot_a << "\n";
        std::cout << "        dot(pos 12,10)= " << dot_b << "  (misma distancia)\n";
        std::cout << "        dot(pos 5,1)  = " << dot_c << "  (otra distancia)\n";

        afirmar("RoPE: misma distancia -> mismo producto punto",
            casi(dot_a, dot_b, 1e-3f));
        afirmar("RoPE: distinta distancia -> distinto producto punto",
            !casi(dot_a, dot_c, 1e-3f));
    }

    // La version de una fila coincide con la version de matriz
    {
        int d = 8;
        Matrix m(1, d);
        std::vector<float> v(d);
        for (int j = 0; j < d; j++) { m.at(0, j) = 0.1f * j - 0.3f; v[j] = m.at(0, j); }

        aplicar_rope(m, 11);
        aplicar_rope_fila(v.data(), d, 11);

        bool bien = true;
        for (int j = 0; j < d; j++) if (!casi(m.at(0, j), v[j])) bien = false;
        afirmar("aplicar_rope_fila == aplicar_rope", bien);
    }

    // Varias filas: cada una usa su propia posicion
    {
        int d = 4;
        Matrix m(3, d);
        for (int i = 0; i < 3; i++)
            for (int j = 0; j < d; j++) m.at(i, j) = 1.0f;

        Matrix esperado(3, d);
        for (int i = 0; i < 3; i++) {
            std::vector<float> v(d, 1.0f);
            aplicar_rope_fila(v.data(), d, 10 + i);   // pos_ini=10
            for (int j = 0; j < d; j++) esperado.at(i, j) = v[j];
        }

        aplicar_rope(m, 10);
        bool bien = true;
        for (int i = 0; i < 3; i++)
            for (int j = 0; j < d; j++)
                if (!casi(m.at(i, j), esperado.at(i, j))) bien = false;
        afirmar("RoPE usa pos_ini + indice de fila", bien);
    }

    // ================= SiLU y SwiGLU =================
    std::cout << "\n--- SiLU / SwiGLU ---\n";

    // SiLU(0) = 0
    afirmar("SiLU(0) = 0", casi(silu(0.0f), 0.0f));
    // SiLU(1) = 1 / (1 + e^-1) = 0.73106
    afirmar("SiLU(1) ~ 0.7311", casi(silu(1.0f), 0.73106f, 1e-4f));
    // SiLU(-1) = -1 / (1 + e) = -0.26894
    afirmar("SiLU(-1) ~ -0.2689", casi(silu(-1.0f), -0.26894f, 1e-4f));
    // Para valores grandes positivos, SiLU(x) ~ x
    afirmar("SiLU(10) ~ 10 (casi identidad)", casi(silu(10.0f), 10.0f, 0.01f));
    // Para negativos grandes, SiLU(x) ~ 0
    afirmar("SiLU(-10) ~ 0 (aplasta negativos)", casi(silu(-10.0f), 0.0f, 0.01f));

    // SwiGLU: forma correcta y la compuerta funciona
    {
        int d = 4, d_ff = 6;
        Matrix x(1, d);
        for (int j = 0; j < d; j++) x.at(0, j) = 0.2f * (j + 1);

        Matrix Wg(d, d_ff), Wu(d, d_ff), Wd(d_ff, d);
        for (int i = 0; i < d * d_ff; i++) {
            Wg.data[i] = 0.05f * (i % 7) - 0.1f;
            Wu.data[i] = 0.04f * (i % 5) + 0.02f;
        }
        for (int i = 0; i < d_ff * d; i++) Wd.data[i] = 0.03f * (i % 9) - 0.05f;

        Matrix r = swiglu(x, Wg, Wu, Wd);
        afirmar("SwiGLU preserva la forma [1 x d]", r.rows == 1 && r.cols == d);
    }

    // Si el gate da cero, la salida es cero: la compuerta cierra
    {
        int d = 2, d_ff = 4;
        Matrix x = { {1, 1} };
        Matrix Wg(d, d_ff);   // todo ceros -> gate = 0 -> SiLU(0) = 0
        Matrix Wu(d, d_ff);
        for (int i = 0; i < d * d_ff; i++) Wu.data[i] = 1.0f;
        Matrix Wd(d_ff, d);
        for (int i = 0; i < d_ff * d; i++) Wd.data[i] = 1.0f;

        Matrix r = swiglu(x, Wg, Wu, Wd);
        bool bien = true;
        for (float v : r.data) if (!casi(v, 0.0f)) bien = false;
        afirmar("SwiGLU: gate en cero cierra la compuerta", bien);
    }

    std::cout << "\n=== " << ok << "/" << total << " pruebas pasaron ===\n";
    if (ok == total)
        std::cout << "RMSNorm, RoPE y SwiGLU listos. Falta GQA y ensamblar.\n";
    return (ok == total) ? 0 : 1;
}