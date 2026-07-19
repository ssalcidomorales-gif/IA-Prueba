// hola_cuda.cu
// Primer programa en CUDA: verifica que la GPU responde y hace la
// operacion mas simple posible en paralelo (sumar dos vectores).
//
// CONCEPTOS BASICOS
//
// KERNEL: una funcion que corre EN LA GPU. Se marca con __global__.
//   Cuando la lanzas, no corre una vez: corre miles de veces en
//   paralelo, una por cada hilo.
//
// LA JERARQUIA DE HILOS
//   kernel<<<bloques, hilos_por_bloque>>>(...)
//
//   Los hilos se agrupan en bloques, y los bloques forman una malla.
//   Cada hilo sabe quien es:
//     threadIdx.x  -> su indice dentro del bloque (0..hilos-1)
//     blockIdx.x   -> a que bloque pertenece
//     blockDim.x   -> cuantos hilos tiene cada bloque
//
//   El indice global de un hilo es:
//     i = blockIdx.x * blockDim.x + threadIdx.x
//
//   Con eso cada hilo decide sobre que dato trabajar.
//
// MEMORIA SEPARADA
//   La GPU tiene su propia RAM. La CPU no puede leerla directamente.
//   Todo dato hay que copiarlo con cudaMemcpy, y los resultados
//   copiarlos de vuelta. Esas copias son LENTAS: la regla de oro es
//   minimizarlas. Por eso mas adelante subiremos los pesos del modelo
//   UNA VEZ y los dejaremos ahi.

#include <cuda_runtime.h>
#include <cstdio>
#include <cmath>
#include <vector>

// ----------------------------------------------------------------------
// Macro para revisar errores de CUDA
// ----------------------------------------------------------------------
// Las llamadas a CUDA no lanzan excepciones: devuelven codigos de error
// que hay que revisar a mano. Sin esto, un fallo pasa silencioso y luego
// aparecen resultados absurdos sin explicacion.
#define CUDA_CHECK(llamada)                                                \
    do {                                                                   \
        cudaError_t err = (llamada);                                       \
        if (err != cudaSuccess) {                                          \
            printf("ERROR CUDA en %s:%d -> %s\n",                          \
                   __FILE__, __LINE__, cudaGetErrorString(err));           \
            return 1;                                                      \
        }                                                                  \
    } while (0)

// ----------------------------------------------------------------------
// EL KERNEL
// ----------------------------------------------------------------------
// Suma dos vectores: c[i] = a[i] + b[i]
//
// Fijate que NO hay bucle. Cada hilo calcula UN elemento. Si lanzas
// 1000 hilos, se calculan 1000 elementos a la vez.
__global__ void sumar_vectores(const float* a, const float* b, float* c, int n) {
    // Cual es mi indice global
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    // Guarda: puede haber mas hilos que datos (los bloques son de tamano
    // fijo, asi que casi siempre sobran algunos hilos al final)
    if (i < n) {
        c[i] = a[i] + b[i];
    }
}

int main() {
    printf("=== Primer programa CUDA ===\n\n");

    // ------------------------------------------------------------------
    // 1. Ver que GPUs hay
    // ------------------------------------------------------------------
    int num_gpus = 0;
    CUDA_CHECK(cudaGetDeviceCount(&num_gpus));

    if (num_gpus == 0) {
        printf("No se encontro ninguna GPU con CUDA.\n");
        return 1;
    }

    printf("GPUs encontradas: %d\n\n", num_gpus);

    for (int i = 0; i < num_gpus; i++) {
        cudaDeviceProp prop;
        CUDA_CHECK(cudaGetDeviceProperties(&prop, i));

        printf("GPU %d: %s\n", i, prop.name);
        printf("  Capacidad de computo: %d.%d\n", prop.major, prop.minor);
        printf("  VRAM total: %.2f GB\n",
               prop.totalGlobalMem / 1024.0 / 1024.0 / 1024.0);
        printf("  Multiprocesadores (SM): %d\n", prop.multiProcessorCount);
        printf("  Max hilos por bloque: %d\n", prop.maxThreadsPerBlock);
        printf("  Memoria compartida por bloque: %.1f KB\n",
               prop.sharedMemPerBlock / 1024.0);
        printf("  Ancho de bus de memoria: %d bits\n", prop.memoryBusWidth);

        // El reloj de memoria se consulta con la API de atributos.
        // (En CUDA 13 quitaron el campo memoryClockRate de cudaDeviceProp)
        int reloj_khz = 0;
        cudaDeviceGetAttribute(&reloj_khz, cudaDevAttrMemoryClockRate, i);
        printf("  Reloj de memoria: %.0f MHz\n", reloj_khz / 1000.0);

        // EL ANCHO DE BANDA es lo que mas importa para inferencia de LLMs:
        // el cuello de botella suele ser LEER los pesos de memoria, no
        // hacer las multiplicaciones. Por eso una GPU con mas ancho de
        // banda genera mas tokens/s aunque tenga menos potencia bruta.
        double ancho_banda = 2.0 * (double)reloj_khz * 1000.0
                           * (prop.memoryBusWidth / 8.0) / 1e9;
        printf("  Ancho de banda teorico: %.1f GB/s\n", ancho_banda);

        // Estimacion practica: cuantas veces por segundo se puede leer
        // un modelo completo de cierto tamano. Es el techo teorico de
        // tokens/s, porque cada token requiere leer TODOS los pesos.
        if (ancho_banda > 0) {
            printf("  Techo teorico con un modelo de 1 GB: %.0f tokens/s\n",
                   ancho_banda / 1.0);
            printf("  Techo teorico con un modelo de 4 GB: %.0f tokens/s\n",
                   ancho_banda / 4.0);
        }
        printf("\n");
    }

    // Memoria libre en este momento
    size_t libre = 0, total = 0;
    CUDA_CHECK(cudaMemGetInfo(&libre, &total));
    printf("VRAM libre ahora: %.2f GB de %.2f GB\n\n",
           libre / 1024.0 / 1024.0 / 1024.0,
           total / 1024.0 / 1024.0 / 1024.0);

    // ------------------------------------------------------------------
    // 2. La primera operacion en GPU
    // ------------------------------------------------------------------
    const int N = 1000000;   // un millon de elementos
    printf("Sumando dos vectores de %d elementos en la GPU...\n", N);

    // Datos en la CPU (memoria "host")
    std::vector<float> h_a(N), h_b(N), h_c(N);
    for (int i = 0; i < N; i++) {
        h_a[i] = (float)i;
        h_b[i] = (float)(i * 2);
    }

    // Punteros a memoria de la GPU (memoria "device")
    float *d_a = nullptr, *d_b = nullptr, *d_c = nullptr;

    // Reservar memoria EN LA GPU
    CUDA_CHECK(cudaMalloc(&d_a, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_b, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_c, N * sizeof(float)));

    // Copiar los datos de CPU a GPU
    CUDA_CHECK(cudaMemcpy(d_a, h_a.data(), N * sizeof(float),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, h_b.data(), N * sizeof(float),
                          cudaMemcpyHostToDevice));

    // Configurar el lanzamiento:
    //   256 hilos por bloque es un valor tipico y eficiente
    //   los bloques necesarios se calculan redondeando hacia arriba
    int hilos_por_bloque = 256;
    int bloques = (N + hilos_por_bloque - 1) / hilos_por_bloque;
    printf("Lanzando %d bloques x %d hilos = %d hilos totales\n",
           bloques, hilos_por_bloque, bloques * hilos_por_bloque);

    // LANZAR EL KERNEL
    sumar_vectores<<<bloques, hilos_por_bloque>>>(d_a, d_b, d_c, N);

    // Revisar si el lanzamiento fallo
    CUDA_CHECK(cudaGetLastError());

    // Esperar a que la GPU termine.
    // Los kernels son ASINCRONOS: la funcion regresa de inmediato y la
    // GPU sigue trabajando en segundo plano. Hay que sincronizar antes
    // de leer los resultados.
    CUDA_CHECK(cudaDeviceSynchronize());

    // Copiar el resultado de vuelta a la CPU
    CUDA_CHECK(cudaMemcpy(h_c.data(), d_c, N * sizeof(float),
                          cudaMemcpyDeviceToHost));

    // ------------------------------------------------------------------
    // 3. Verificar
    // ------------------------------------------------------------------
    bool correcto = true;
    for (int i = 0; i < N; i++) {
        float esperado = h_a[i] + h_b[i];
        if (std::fabs(h_c[i] - esperado) > 1e-3f) {
            printf("ERROR en el indice %d: obtuve %f, esperaba %f\n",
                   i, h_c[i], esperado);
            correcto = false;
            break;
        }
    }

    printf("\nPrimeros resultados: ");
    for (int i = 0; i < 5; i++) printf("%.0f ", h_c[i]);
    printf("...\n");

    if (correcto)
        printf("\nTODOS LOS RESULTADOS CORRECTOS.\n");
    else
        printf("\nHubo errores en el resultado.\n");

    // Liberar la memoria de la GPU
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    printf("\n=== El entorno CUDA funciona ===\n");
    return correcto ? 0 : 1;
}
