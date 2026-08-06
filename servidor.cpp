// servidor.cpp
// Servidor TCP que expone el motor Qwen a clientes externos (WinForms,
// linea de comandos, lo que sea).
//
// POR QUE UN SERVIDOR Y NO TODO EN UNA APP
//   - El modelo se carga UNA VEZ al arrancar, no en cada consulta.
//     Cargar 900 MB toma varios segundos; hacerlo por mensaje seria
//     inaceptable.
//   - El motor se queda en C++, donde ya esta y donde rinde.
//   - Cualquier cliente puede conectarse: WinForms, web, consola.
//
// EL PROTOCOLO (texto plano, una linea por mensaje)
//
//   Cliente -> Servidor:
//     CHAT <mensaje del usuario>      genera una respuesta
//     SISTEMA <texto>                 cambia el prompt de sistema
//     LIMPIAR                         borra el historial
//     CONFIG <clave> <valor>          ajusta temperatura, top_k, etc
//     PING                            prueba de vida
//     SALIR                           cierra la conexion
//
//   Servidor -> Cliente:
//     LISTO <info del modelo>         al conectar
//     TOKEN <texto>                   un token generado (streaming)
//     FIN <tokens> <segundos>         termino de generar
//     OK <mensaje>                    comando ejecutado
//     ERROR <mensaje>                 algo salio mal
//     PONG                            respuesta a PING
//
// Los saltos de linea dentro de los tokens se escapan como \n para que
// cada mensaje quepa en una sola linea.

#include "qwen.h"
#include "sampling.h"

#include <iostream>
#include <string>
#include <vector>
#include <sstream>
#include <chrono>
#include <thread>
#include <atomic>

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#include <winsock2.h>
#include <ws2tcpip.h>
#pragma comment(lib, "ws2_32.lib")
typedef SOCKET socket_t;
#define CERRAR_SOCKET closesocket
#else
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <unistd.h>
typedef int socket_t;
#define INVALID_SOCKET (-1)
#define SOCKET_ERROR (-1)
#define CERRAR_SOCKET close
#endif

// ----------------------------------------------------------------------
// Utilidades de texto
// ----------------------------------------------------------------------

// Escapa saltos de linea para que un token quepa en una sola linea
// del protocolo. El cliente los desescapa.
std::string escapar(const std::string& s) {
    std::string r;
    r.reserve(s.size() + 8);
    for (char c : s) {
        if (c == '\\') r += "\\\\";
        else if (c == '\n') r += "\\n";
        else if (c == '\r') r += "\\r";
        else r += c;
    }
    return r;
}

std::string desescapar(const std::string& s) {
    std::string r;
    r.reserve(s.size());
    for (size_t i = 0; i < s.size(); i++) {
        if (s[i] == '\\' && i + 1 < s.size()) {
            char c = s[++i];
            if (c == 'n')  r += '\n';
            else if (c == 'r')  r += '\r';
            else if (c == '\\') r += '\\';
            else r += c;
        }
        else {
            r += s[i];
        }
    }
    return r;
}

// ----------------------------------------------------------------------
// Un turno de la conversacion
// ----------------------------------------------------------------------
struct Turno {
    std::string rol;      // "user" o "assistant"
    std::string texto;
};

// ----------------------------------------------------------------------
// Sesion: el estado de un cliente conectado
// ----------------------------------------------------------------------
// Cada cliente tiene su propio historial y configuracion, asi que dos
// conversaciones distintas no se mezclan.
struct Sesion {
    std::vector<Turno> historial;
    std::string sistema = "Eres un asistente util. Respondes de forma "
        "clara y concisa en el idioma del usuario.";
    ConfigSampling cfg;
    int max_tokens = 400;

    // Cuantos turnos conservar. Si la conversacion crece mucho, los
    // mensajes viejos se descartan para no rebasar el contexto.
    int max_turnos = 20;

    void agregar(const std::string& rol, const std::string& texto) {
        historial.push_back({ rol, texto });
        while ((int)historial.size() > max_turnos)
            historial.erase(historial.begin());
    }

    void limpiar() { historial.clear(); }

    // Arma el prompt completo en formato ChatML, incluyendo todo el
    // historial. Asi el modelo "recuerda" la conversacion: no hay magia,
    // simplemente se le pasa todo el contexto en cada consulta.
    std::string construir_prompt() const {
        std::string p;
        if (!sistema.empty())
            p += "<|im_start|>system\n" + sistema + "<|im_end|>\n";
        for (const Turno& t : historial)
            p += "<|im_start|>" + t.rol + "\n" + t.texto + "<|im_end|>\n";
        p += "<|im_start|>assistant\n";
        return p;
    }
};

// ----------------------------------------------------------------------
// Envio por socket
// ----------------------------------------------------------------------
bool enviar_linea(socket_t s, const std::string& linea) {
    std::string datos = linea + "\n";
    size_t enviados = 0;
    while (enviados < datos.size()) {
        int n = send(s, datos.data() + enviados,
            (int)(datos.size() - enviados), 0);
        if (n <= 0) return false;
        enviados += n;
    }
    return true;
}

// Lee una linea completa del socket (hasta el \n)
bool leer_linea(socket_t s, std::string& linea, std::string& buffer) {
    // Primero ver si ya hay una linea completa en el buffer
    size_t pos = buffer.find('\n');
    while (pos == std::string::npos) {
        char temp[4096];
        int n = recv(s, temp, sizeof(temp), 0);
        if (n <= 0) return false;   // conexion cerrada o error
        buffer.append(temp, n);
        pos = buffer.find('\n');
    }
    linea = buffer.substr(0, pos);
    buffer.erase(0, pos + 1);
    // Quitar el \r si viene de un cliente Windows
    if (!linea.empty() && linea.back() == '\r') linea.pop_back();
    return true;
}

// ----------------------------------------------------------------------
// Atender a un cliente
// ----------------------------------------------------------------------
void atender_cliente(socket_t cliente, Qwen& modelo) {
    Sesion sesion;
    Sampler sampler;
    std::string buffer;

    // Saludo con informacion del modelo
    {
        std::ostringstream ss;
        ss << "LISTO capas=" << modelo.num_capas
            << " d_model=" << modelo.d_model
            << " vocab=" << modelo.vocab
            << " contexto=" << modelo.contexto_max;
        enviar_linea(cliente, ss.str());
    }

    std::string linea;
    while (leer_linea(cliente, linea, buffer)) {
        if (linea.empty()) continue;

        // Separar el comando del resto
        size_t esp = linea.find(' ');
        std::string comando = linea.substr(0, esp);
        std::string resto = (esp == std::string::npos)
            ? "" : linea.substr(esp + 1);

        // ---------------- PING ----------------
        if (comando == "PING") {
            enviar_linea(cliente, "PONG");
            continue;
        }

        // ---------------- SALIR ----------------
        if (comando == "SALIR") {
            enviar_linea(cliente, "OK hasta luego");
            break;
        }

        // ---------------- LIMPIAR ----------------
        if (comando == "LIMPIAR") {
            sesion.limpiar();
            enviar_linea(cliente, "OK historial borrado");
            continue;
        }

        // ---------------- SISTEMA ----------------
        if (comando == "SISTEMA") {
            sesion.sistema = desescapar(resto);
            enviar_linea(cliente, "OK prompt de sistema actualizado");
            continue;
        }

        // ---------------- CONFIG ----------------
        if (comando == "CONFIG") {
            std::istringstream ss(resto);
            std::string clave, valor;
            ss >> clave >> valor;
            try {
                if (clave == "temp")      sesion.cfg.temperatura = std::stof(valor);
                else if (clave == "topk")      sesion.cfg.top_k = std::stoi(valor);
                else if (clave == "topp")      sesion.cfg.top_p = std::stof(valor);
                else if (clave == "penal")     sesion.cfg.penal_repeticion = std::stof(valor);
                else if (clave == "max")       sesion.max_tokens = std::stoi(valor);
                else if (clave == "turnos")    sesion.max_turnos = std::stoi(valor);
                else {
                    enviar_linea(cliente, "ERROR clave desconocida: " + clave);
                    continue;
                }
                enviar_linea(cliente, "OK " + clave + "=" + valor);
            }
            catch (...) {
                enviar_linea(cliente, "ERROR valor invalido");
            }
            continue;
        }

        // ---------------- CHAT ----------------
        if (comando == "CHAT") {
            std::string mensaje = desescapar(resto);
            if (mensaje.empty()) {
                enviar_linea(cliente, "ERROR mensaje vacio");
                continue;
            }

            sesion.agregar("user", mensaje);
            std::string prompt_texto = sesion.construir_prompt();
            std::vector<int> prompt = modelo.codificar(prompt_texto);

            // Si el prompt ya es enorme, recortar turnos viejos
            while ((int)prompt.size() > modelo.contexto_max - sesion.max_tokens - 64
                && sesion.historial.size() > 2) {
                sesion.historial.erase(sesion.historial.begin());
                prompt = modelo.codificar(sesion.construir_prompt());
            }

            auto cache = modelo.crear_cache();
            sampler.reiniciar();

            auto t1 = std::chrono::high_resolution_clock::now();
            Matrix logits;
            try {
                logits = modelo.procesar_prompt(prompt, cache);
            }
            catch (const std::exception& e) {
                enviar_linea(cliente, std::string("ERROR ") + e.what());
                continue;
            }

            std::string respuesta;
            int generados = 0;
            bool conexion_viva = true;

            for (int i = 0; i < sesion.max_tokens; i++) {
                int sig = sampler.elegir(logits, sesion.cfg);

                if (sig == modelo.eos ||
                    sig == modelo.bpe.id_de("<|im_end|>"))
                    break;

                std::string txt = modelo.decodificar_uno(sig);
                respuesta += txt;
                generados++;

                // Enviar el token al cliente de inmediato (streaming)
                if (!enviar_linea(cliente, "TOKEN " + escapar(txt))) {
                    conexion_viva = false;
                    break;
                }

                if (cache.tokens_procesados() >= modelo.contexto_max - 1) break;

                try {
                    logits = modelo.forward(sig, cache.tokens_procesados(), cache);
                }
                catch (const std::exception& e) {
                    enviar_linea(cliente, std::string("ERROR ") + e.what());
                    conexion_viva = false;
                    break;
                }
            }

            if (!conexion_viva) break;

            auto t2 = std::chrono::high_resolution_clock::now();
            double seg = std::chrono::duration<double>(t2 - t1).count();

            // Guardar la respuesta en el historial para el siguiente turno
            sesion.agregar("assistant", respuesta);

            std::ostringstream ss;
            ss << "FIN " << generados << " " << seg
                << " prompt=" << prompt.size();
            enviar_linea(cliente, ss.str());
            continue;
        }

        // --- INYECTAR <rol> <texto> : mete un turno al historial sin generar ---
        // Sirve para rehidratar la sesion cuando el cliente abre una
        // conversacion vieja guardada en SQLite.
        if (comando == "INYECTAR") {
            std::istringstream entrada(resto);
            std::string rol;
            entrada >> rol;

            std::string texto;
            std::getline(entrada, texto);
            if (!texto.empty() && texto[0] == ' ') texto.erase(0, 1);

            // Un rol invalido corrompe el prompt ChatML y el modelo empieza a
            // responderse a si mismo. Mejor rechazarlo aqui.
            if (rol != "user" && rol != "assistant") {
                enviar_linea(cliente, "ERROR rol invalido: " + rol);
                continue;
            }
            if (texto.empty()) {
                enviar_linea(cliente, "ERROR texto vacio");
                continue;
            }

            sesion.agregar(rol, desescapar(texto));
            enviar_linea(cliente, "OK inyectado");
            continue;
        }

        enviar_linea(cliente, "ERROR comando desconocido: " + comando);
    }

    CERRAR_SOCKET(cliente);
    std::cout << "[servidor] cliente desconectado\n";
}

// ----------------------------------------------------------------------
int main(int argc, char** argv) {
    std::string ruta =
        "C:/Users/Said/source/repos/IA_Prueba/IA_Prueba/qwen2.5-1.5b-instruct-q4_0.gguf";
    int puerto = 8420;

    if (argc >= 2) ruta = argv[1];
    if (argc >= 3) puerto = std::atoi(argv[2]);

    std::cout << "=== Servidor Jarvis ===\n\n";

    // --- Cargar el modelo (una sola vez) ---
    Qwen modelo;
    try {
        modelo.cargar(ruta);
    }
    catch (const std::exception& e) {
        std::cout << "ERROR cargando el modelo: " << e.what() << "\n";
        return 1;
    }

    if (!modelo.bpe.listo()) {
        std::cout << "ERROR: el tokenizador BPE no se construyo.\n";
        return 1;
    }

#ifdef _WIN32
    WSADATA wsa;
    if (WSAStartup(MAKEWORD(2, 2), &wsa) != 0) {
        std::cout << "ERROR: no pude inicializar Winsock\n";
        return 1;
    }
#endif

    // --- Crear el socket de escucha ---
    socket_t servidor = socket(AF_INET, SOCK_STREAM, 0);
    if (servidor == INVALID_SOCKET) {
        std::cout << "ERROR: no pude crear el socket\n";
        return 1;
    }

    // Permitir reusar el puerto de inmediato tras cerrar (evita el
    // clasico "address already in use" al reiniciar el servidor)
    int si = 1;
    setsockopt(servidor, SOL_SOCKET, SO_REUSEADDR, (const char*)&si, sizeof(si));

    sockaddr_in dir{};
    dir.sin_family = AF_INET;
    dir.sin_port = htons((unsigned short)puerto);
    // Solo local: nadie fuera de esta maquina puede conectarse.
    // inet_pton es la API moderna (inet_addr esta deprecada).
    inet_pton(AF_INET, "127.0.0.1", &dir.sin_addr);

    if (bind(servidor, (sockaddr*)&dir, sizeof(dir)) == SOCKET_ERROR) {
        std::cout << "ERROR: no pude enlazar al puerto " << puerto << "\n";
        std::cout << "Puede que ya haya otro servidor corriendo.\n";
        CERRAR_SOCKET(servidor);
        return 1;
    }

    if (listen(servidor, 4) == SOCKET_ERROR) {
        std::cout << "ERROR: no pude escuchar\n";
        CERRAR_SOCKET(servidor);
        return 1;
    }

    std::cout << "\nEscuchando en 127.0.0.1:" << puerto << "\n";
    std::cout << "Esperando conexiones... (Ctrl+C para salir)\n\n";

    // --- Bucle de aceptacion ---
    // Atendemos un cliente a la vez: el modelo no es thread-safe y de
    // todos modos la GPU/CPU solo puede generar una respuesta a la vez.
    while (true) {
        sockaddr_in dir_cliente{};
        int tam = sizeof(dir_cliente);
#ifdef _WIN32
        socket_t cliente = accept(servidor, (sockaddr*)&dir_cliente, &tam);
#else
        socklen_t tam2 = tam;
        socket_t cliente = accept(servidor, (sockaddr*)&dir_cliente, &tam2);
#endif
        if (cliente == INVALID_SOCKET) continue;

        std::cout << "[servidor] cliente conectado\n";
        atender_cliente(cliente, modelo);
    }

    CERRAR_SOCKET(servidor);
#ifdef _WIN32
    WSACleanup();
#endif
    return 0;
}