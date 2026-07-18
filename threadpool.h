// threadpool.h
// FASE 1 - Optimizacion: pool de hilos.
//
// EL PROBLEMA
// Crear un hilo cuesta del orden de 1-3 ms en Windows. El feed-forward
// tarda ~8 ms, asi que creando 16 hilos pagabamos casi la mitad del
// trabajo en pura administracion. Y eso pasa 48 veces por token
// (4 matmuls por bloque x 12 bloques).
//
// LA IDEA
// Crear los hilos UNA SOLA VEZ al arrancar y mantenerlos vivos esperando
// trabajo. Cuando hay una tarea, se les avisa; cuando terminan, vuelven
// a dormir. El costo de creacion se paga una vez, no en cada llamada.
//
// COMO FUNCIONA
// Cada hilo corre un bucle: espera una senal, hace su parte del trabajo,
// avisa que termino, vuelve a esperar. La sincronizacion usa un mutex y
// variables de condicion, que es la forma estandar de coordinar hilos
// sin quemar CPU en espera activa.

#ifndef THREADPOOL_H
#define THREADPOOL_H

#include <thread>
#include <vector>
#include <mutex>
#include <condition_variable>
#include <functional>
#include <atomic>
#include <algorithm>

class ThreadPool {
public:
    // Crea el pool con n hilos. Se llama una vez al inicio del programa.
    explicit ThreadPool(int n_hilos) : parar_(false), pendientes_(0) {
        n_hilos = std::max(1, n_hilos);
        hilos_.reserve(n_hilos);
        for (int i = 0; i < n_hilos; i++)
            hilos_.emplace_back([this, i] { bucle_trabajador(i); });
    }

    ~ThreadPool() {
        {
            std::unique_lock<std::mutex> lock(mtx_);
            parar_ = true;
        }
        cv_trabajo_.notify_all();
        for (auto& t : hilos_) if (t.joinable()) t.join();
    }

    // No se puede copiar ni mover: administra recursos del sistema
    ThreadPool(const ThreadPool&) = delete;
    ThreadPool& operator=(const ThreadPool&) = delete;

    int tamano() const { return (int)hilos_.size(); }

    // Ejecuta 'tarea' en paralelo y espera a que todos terminen.
    // La tarea recibe el indice del hilo (0, 1, 2, ...) para que sepa
    // que porcion del trabajo le toca.
    void ejecutar(const std::function<void(int)>& tarea) {
        {
            std::unique_lock<std::mutex> lock(mtx_);
            tarea_ = tarea;
            pendientes_ = (int)hilos_.size();
            generacion_++;
        }
        cv_trabajo_.notify_all();

        // Esperar a que todos terminen
        std::unique_lock<std::mutex> lock(mtx_);
        cv_listo_.wait(lock, [this] { return pendientes_ == 0; });
    }

private:
    void bucle_trabajador(int id) {
        unsigned long long mi_generacion = 0;
        while (true) {
            std::function<void(int)> mi_tarea;
            {
                std::unique_lock<std::mutex> lock(mtx_);
                // Dormir hasta que haya trabajo nuevo o toque parar
                cv_trabajo_.wait(lock, [this, &mi_generacion] {
                    return parar_ || generacion_ != mi_generacion;
                });
                if (parar_) return;
                mi_generacion = generacion_;
                mi_tarea = tarea_;
            }

            // Trabajar fuera del lock: aqui es donde ocurre el paralelismo real
            mi_tarea(id);

            {
                std::unique_lock<std::mutex> lock(mtx_);
                if (--pendientes_ == 0) cv_listo_.notify_all();
            }
        }
    }

    std::vector<std::thread> hilos_;
    std::mutex mtx_;
    std::condition_variable cv_trabajo_;   // avisa que hay tarea nueva
    std::condition_variable cv_listo_;     // avisa que todos terminaron
    std::function<void(int)> tarea_;
    bool parar_;
    int pendientes_;
    unsigned long long generacion_ = 0;    // distingue tareas sucesivas
};

#endif // THREADPOOL_H
