// sampling.h
// Estrategias de muestreo con penalizacion por repeticion.
//
// EL PROBLEMA DE LOS BUCLES
// Un modelo chico tiende a repetirse: genera "Foros de futbol", y como
// ese texto ahora esta en el contexto, lo mas probable segun el modelo
// es... volver a generar "Foros de futbol". Se retroalimenta y entra en
// un ciclo del que no sale.
//
// LA SOLUCION
// Penalizar los tokens que ya aparecieron recientemente: bajarles el
// logit antes de muestrear. Cuantas mas veces aparecio un token, menos
// probable se vuelve. Eso empuja al modelo a buscar alternativas.
//
// EL DETALLE DEL SIGNO
// Un logit puede ser positivo o negativo, asi que no basta con dividir:
//   - si el logit es positivo, dividir lo hace mas chico (lo penaliza)
//   - si es negativo, dividir lo hace MENOS negativo (lo premia!)
// Por eso se multiplica cuando es negativo y se divide cuando es positivo.

#ifndef SAMPLING_H
#define SAMPLING_H

#include "matrix.h"
#include <vector>
#include <deque>
#include <random>
#include <algorithm>
#include <cmath>
#include <unordered_set>

// Configuracion de muestreo
struct ConfigSampling {
    float temperatura = 0.7f;      // <=0 significa greedy
    int   top_k = 40;              // 0 = sin limite
    float top_p = 0.95f;           // muestreo por nucleo; >=1 lo desactiva
    float penal_repeticion = 1.1f; // 1.0 = sin penalizacion
    int   ventana_repeticion = 64; // cuantos tokens recientes considerar
};

class Sampler {
public:
    explicit Sampler(unsigned semilla = std::random_device{}())
        : rng_(semilla) {}

    void reiniciar() { recientes_.clear(); }

    // Registra un token generado, para la penalizacion futura
    void registrar(int token, int ventana) {
        recientes_.push_back(token);
        while ((int)recientes_.size() > ventana) recientes_.pop_front();
    }

    // Elige el siguiente token
    int elegir(const Matrix& logits, const ConfigSampling& cfg) {
        int V = logits.cols;

        // Copiamos los logits para poder modificarlos
        std::vector<float> l(V);
        for (int j = 0; j < V; j++) l[j] = logits.at(0, j);

        // --- 1. Penalizacion por repeticion ---
        //
        // EL DETALLE DEL SIGNO
        // Lo intuitivo seria dividir el logit por la penalizacion, pero
        // eso falla con logits negativos: dividir -1 entre 2 da -0.5,
        // que es MAYOR. Estarias premiando la repeticion.
        //
        // Multiplicar los negativos tampoco basta: si tienes [-1, -2] y
        // penalizas el -1 con factor 2, queda -2, que EMPATA con el otro
        // y el greedy se queda pegado en el primero.
        //
        // La solucion: restar una cantidad proporcional. Asi el token
        // penalizado siempre baja de verdad, sin importar su signo.
        if (cfg.penal_repeticion != 1.0f && !recientes_.empty()) {
            std::unordered_set<int> vistos(recientes_.begin(), recientes_.end());

            // Cuanto restar: escalado con el rango de los logits para que
            // la penalizacion sea significativa sin importar la magnitud
            float mx = l[0], mn = l[0];
            for (int j = 1; j < V; j++) {
                if (l[j] > mx) mx = l[j];
                if (l[j] < mn) mn = l[j];
            }
            // El castigo debe SUPERAR el rango, no solo igualarlo: si
            // apenas lo iguala, el token penalizado puede quedar empatado
            // con otro y el greedy se queda pegado en el primero.
            float rango = std::max(1.0f, mx - mn);
            float castigo = (cfg.penal_repeticion - 1.0f) * rango * 1.5f
                            + 1e-3f;

            for (int t : vistos) {
                if (t >= 0 && t < V) l[t] -= castigo;
            }
        }

        // --- 2. Greedy: el mayor y listo ---
        if (cfg.temperatura <= 0.0f) {
            int mejor = 0;
            for (int j = 1; j < V; j++) if (l[j] > l[mejor]) mejor = j;
            registrar(mejor, cfg.ventana_repeticion);
            return mejor;
        }

        // --- 3. Temperatura ---
        for (int j = 0; j < V; j++) l[j] /= cfg.temperatura;

        // --- 4. Top-k: quedarse con los k mejores ---
        std::vector<std::pair<float,int>> cand;
        cand.reserve(V);
        for (int j = 0; j < V; j++) cand.push_back({l[j], j});

        int k = (cfg.top_k > 0 && cfg.top_k < V) ? cfg.top_k : V;
        std::partial_sort(cand.begin(), cand.begin()+k, cand.end(),
                          [](const auto& a, const auto& b){
                              return a.first > b.first;
                          });
        cand.resize(k);

        // --- 5. Softmax sobre los candidatos ---
        float mx = cand[0].first;
        float suma = 0.0f;
        for (auto& p : cand) { p.first = std::exp(p.first - mx); suma += p.first; }
        for (auto& p : cand) p.first /= suma;

        // --- 6. Top-p (nucleo): quedarse con los que acumulan probabilidad p ---
        // La idea: en vez de un numero fijo de candidatos, tomar los
        // suficientes para cubrir el p% de la probabilidad total. Si el
        // modelo esta muy seguro, seran pocos; si duda, seran mas.
        if (cfg.top_p < 1.0f) {
            float acum = 0.0f;
            size_t corte = cand.size();
            for (size_t i = 0; i < cand.size(); i++) {
                acum += cand[i].first;
                if (acum >= cfg.top_p) { corte = i + 1; break; }
            }
            cand.resize(corte);
            // renormalizar
            float s = 0.0f;
            for (auto& p : cand) s += p.first;
            for (auto& p : cand) p.first /= s;
        }

        // --- 7. Muestrear ---
        std::uniform_real_distribution<float> dist(0.0f, 1.0f);
        float r = dist(rng_);
        float acum = 0.0f;
        int elegido = cand.back().second;
        for (auto& p : cand) {
            acum += p.first;
            if (r <= acum) { elegido = p.second; break; }
        }

        registrar(elegido, cfg.ventana_repeticion);
        return elegido;
    }

private:
    std::mt19937 rng_;
    std::deque<int> recientes_;
};

#endif // SAMPLING_H
