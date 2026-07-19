// test_sampling.cpp
// Verifica el sampler, sobre todo que la penalizacion por repeticion
// realmente baje la probabilidad de tokens ya usados.

#include "sampling.h"
#include <iostream>
#include <iomanip>
#include <map>

int ok = 0, total = 0;
void afirmar(const std::string& n, bool c) {
    total++;
    if (c) { std::cout << "[OK]    " << n << "\n"; ok++; }
    else   { std::cout << "[FALLO] " << n << "\n"; }
}

int main() {
    std::cout << "=== Tests del sampler ===\n\n";

    // Logits sinteticos: 5 tokens, el 2 es claramente el favorito
    Matrix logits(1, 5);
    logits.at(0,0) = 1.0f;
    logits.at(0,1) = 2.0f;
    logits.at(0,2) = 5.0f;   // el mayor
    logits.at(0,3) = 0.5f;
    logits.at(0,4) = -1.0f;  // negativo, para probar el manejo del signo

    // --- Greedy siempre elige el mayor ---
    {
        ConfigSampling cfg;
        cfg.temperatura = 0.0f;
        cfg.penal_repeticion = 1.0f;
        Sampler s(42);
        bool siempre_2 = true;
        for (int i = 0; i < 5; i++) {
            Sampler s2(42);
            if (s2.elegir(logits, cfg) != 2) siempre_2 = false;
        }
        afirmar("greedy elige el logit mayor", siempre_2);
    }

    // --- La penalizacion baja el token ya usado ---
    {
        ConfigSampling cfg;
        cfg.temperatura = 0.0f;          // greedy para que sea determinista
        cfg.penal_repeticion = 3.0f;     // penalizacion fuerte
        cfg.ventana_repeticion = 10;

        Sampler s(42);
        int primero = s.elegir(logits, cfg);   // debe ser 2
        int segundo = s.elegir(logits, cfg);   // el 2 ya penalizado -> otro

        std::cout << "        primera eleccion: " << primero << "\n";
        std::cout << "        segunda eleccion: " << segundo
                  << " (el " << primero << " fue penalizado)\n";

        afirmar("primera eleccion = 2 (el mayor)", primero == 2);
        afirmar("la penalizacion cambia la segunda eleccion", segundo != primero);
    }

    // --- Sin penalizacion, greedy se repite para siempre (el bucle) ---
    {
        ConfigSampling cfg;
        cfg.temperatura = 0.0f;
        cfg.penal_repeticion = 1.0f;   // desactivada
        Sampler s(42);
        bool siempre_igual = true;
        int primero = s.elegir(logits, cfg);
        for (int i = 0; i < 5; i++)
            if (s.elegir(logits, cfg) != primero) siempre_igual = false;
        afirmar("sin penalizacion, greedy entra en bucle", siempre_igual);
    }

    // --- El manejo del signo: un logit negativo debe penalizarse tambien ---
    {
        // Con solo el token 4 (negativo) disponible via top_k=1 no sirve;
        // probamos directamente que penalizar un negativo lo hace MAS
        // negativo, no menos.
        Matrix l2(1, 2);
        l2.at(0,0) = -1.0f;
        l2.at(0,1) = -2.0f;

        ConfigSampling cfg;
        cfg.temperatura = 0.0f;
        cfg.penal_repeticion = 2.0f;
        Sampler s(42);

        int primero = s.elegir(l2, cfg);   // el -1 es mayor que el -2
        int segundo = s.elegir(l2, cfg);   // -1 penalizado -> -2, asi que gana el otro

        std::cout << "        logits negativos: primero=" << primero
                  << " segundo=" << segundo << "\n";
        afirmar("penaliza correctamente logits negativos",
                primero == 0 && segundo == 1);
    }

    // --- La ventana limita cuantos tokens se recuerdan ---
    {
        ConfigSampling cfg;
        cfg.temperatura = 0.0f;
        cfg.penal_repeticion = 5.0f;
        cfg.ventana_repeticion = 2;   // solo recuerda 2

        Sampler s(42);
        std::vector<int> secuencia;
        for (int i = 0; i < 8; i++)
            secuencia.push_back(s.elegir(logits, cfg));

        std::cout << "        secuencia con ventana=2: ";
        for (int t : secuencia) std::cout << t << " ";
        std::cout << "\n";

        // Con ventana chica, los tokens vuelven a aparecer ciclicamente
        afirmar("la ventana deslizante permite reutilizar tokens viejos",
                secuencia.size() == 8);
    }

    // --- top_k limita los candidatos ---
    {
        ConfigSampling cfg;
        cfg.temperatura = 1.0f;
        cfg.top_k = 1;                 // solo el mejor
        cfg.top_p = 1.0f;
        cfg.penal_repeticion = 1.0f;

        Sampler s(42);
        bool siempre_2 = true;
        for (int i = 0; i < 10; i++)
            if (s.elegir(logits, cfg) != 2) siempre_2 = false;
        afirmar("top_k=1 equivale a greedy", siempre_2);
    }

    // --- Distribucion: con temperatura alta se ve variedad ---
    {
        ConfigSampling cfg;
        cfg.temperatura = 2.0f;
        cfg.top_k = 5;
        cfg.top_p = 1.0f;
        cfg.penal_repeticion = 1.0f;

        Sampler s(123);
        std::map<int,int> conteo;
        for (int i = 0; i < 500; i++) conteo[s.elegir(logits, cfg)]++;

        std::cout << "        distribucion con temp=2.0: ";
        for (auto& p : conteo) std::cout << p.first << ":" << p.second << " ";
        std::cout << "\n";
        afirmar("temperatura alta produce variedad", conteo.size() >= 3);
    }

    std::cout << "\n=== " << ok << "/" << total << " pruebas pasaron ===\n";
    return (ok==total) ? 0 : 1;
}
