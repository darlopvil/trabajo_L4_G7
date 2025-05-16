#include <stdio.h>
#include <math.h>
#include <omp.h>
#include <stdlib.h>
#include <time.h>
#include <fstream> // Para manejo de archivos

// Estructura para almacenar los resultados
struct ResultadoMontecarlo {
    double pi;
    double tiempo_segundos;
    double tiempo_ms;
    double tiempo_us;
    long long samples;
    bool es_paralelo;
    int num_hilos; // Solo relevante para la versión paralela
};

// SIN PARALELIZAR - METODO DE MONTECARLO
ResultadoMontecarlo montecarlo_secuencial(long long samples) {
    unsigned long long count = 0;
    unsigned long long i;
    double x, y;
    double inicio, final, total = 0;
    ResultadoMontecarlo resultado;
    
    resultado.samples = samples;
    resultado.es_paralelo = false;
    resultado.num_hilos = 1;

    inicio = omp_get_wtime();

    for (i = 0; i < samples; ++i) {
        //Crear un punto para tirar en el cuadrante
        x = ((double)rand()) / ((double)RAND_MAX);  //0 <= x <= 1
        y = ((double)rand()) / ((double)RAND_MAX);
        //Si el punto cae dentro del círculo, incrementar el contador
        if (x * x + y * y <= 1.0) {
            ++count;
        }
    }

    final = omp_get_wtime();
    total = (final - inicio);
    
    // Guardar resultados
    resultado.pi = 4.0 * count / samples;
    resultado.tiempo_segundos = total;
    resultado.tiempo_ms = total * 1e3;
    resultado.tiempo_us = total * 1e6;

    printf("----------------OpenMP MonterCarlo Sin Paralelizar----------------\n");
    printf("Numero de Samples = %lld\n", samples);
    printf("pi = %.12f\n", resultado.pi);  //4 cuadrantes, solo se calcula en 1
    printf("Tiempo de ejec./elemento de calculo (en segundos) => %.12lf s\n", resultado.tiempo_segundos);
    printf("Tiempo de ejec./elemento de calculo (en milisegundos) => %.8lf ms\n", resultado.tiempo_ms);
    printf("Tiempo de ejec./elemento de calculo (en microsegundos) => %.8lf us\n", resultado.tiempo_us);
    printf("-------------------------------------------------------------------\n\n");

    return resultado;
}

// PARALELIZADO - METODO DE MONTECARLO
ResultadoMontecarlo montecarlo_paralelo(long long samples) {
    
    unsigned long long count = 0;
    long long i, a;
    double x, y;
    double inicio, final, total = 0;
    ResultadoMontecarlo resultado;
    
    resultado.samples = samples;
    resultado.es_paralelo = true;

    a = omp_get_num_procs();
    int num_threads = 8; // Número de hilos disponibles
    omp_set_num_threads(num_threads);
    resultado.num_hilos = num_threads;

    // Crear array de semillas con tamaño adecuado
    unsigned short* xi = new unsigned short[num_threads];
    srand(time(NULL)); // Semilla base basada en tiempo actual
    for (int j = 0; j < num_threads; j++) {
        xi[j] = rand() % 10000 + 1; // Generar semillas aleatorias diferentes
    }

    inicio = omp_get_wtime();

#pragma omp parallel private(x,y)
    {
        // Semilla diferente para cada hilo
        int tid = omp_get_thread_num();
        srand(xi[tid]);

#pragma omp for reduction(+:count)
        for (i = 0; i < samples; ++i) {
            x = ((double)rand()) / ((double)RAND_MAX);
            y = ((double)rand()) / ((double)RAND_MAX);
            if (x * x + y * y <= 1.0) {
                ++count;
            }
        }
    }

    final = omp_get_wtime();
    total = (final - inicio);
    
    // Guardar resultados
    resultado.pi = 4.0 * count / samples;
    resultado.tiempo_segundos = total;
    resultado.tiempo_ms = total * 1e3;
    resultado.tiempo_us = total * 1e6;

    printf("----------------OpenMP MonterCarlo Paralelizado----------------\n");
    printf("Numero de Procesadores: %lld\n", a);
    printf("Numero de Hilos utilizados: %d\n", num_threads);
    printf("Numero de Samples = %lld\n", samples);
    printf("pi = %.12f\n", resultado.pi);
    printf("Tiempo de ejec./elemento de calculo (en segundos) => %.12lf s\n", resultado.tiempo_segundos);
    printf("Tiempo de ejec./elemento de calculo (en milisegundos) => %.8lf ms\n", resultado.tiempo_ms);
    printf("Tiempo de ejec./elemento de calculo (en microsegundos) => %.8lf us\n", resultado.tiempo_us);
    printf("-------------------------------------------------------------------\n\n");

    // Liberar memoria
    delete[] xi;
    
    return resultado;
}

// Función para guardar los resultados en un archivo CSV
void guardar_csv(const ResultadoMontecarlo& secuencial, const ResultadoMontecarlo& paralelo, const char* nombre_archivo) {
    std::ofstream archivo(nombre_archivo);

    if (!archivo.is_open()) {
        printf("Error: No se pudo abrir el archivo %s para escritura\n", nombre_archivo);
        return;
    }

    // Usar punto y coma como separador y coma como decimal
    archivo << "Metodo;Samples;Hilos;Valor Pi;Tiempo (s);Tiempo (ms);Tiempo (us)\n";

    // Función para formatear números con coma decimal y precisión específica
    auto formatearDecimal = [](double valor, int precision) -> std::string {
        char buffer[64];
        // Usar snprintf para controlar la precisión decimal
        snprintf(buffer, sizeof(buffer), "%.*f", precision, valor);
        std::string resultado = buffer;
        // Reemplazar puntos por comas para el formato español
        for (char& c : resultado) {
            if (c == '.') c = ',';
        }
        return resultado;
        };

    // Escribir datos secuenciales
    archivo << "Secuencial;" << secuencial.samples << ";" << secuencial.num_hilos << ";"
        << formatearDecimal(secuencial.pi, 12) << ";"
        << formatearDecimal(secuencial.tiempo_segundos, 12) << ";"
        << formatearDecimal(secuencial.tiempo_ms, 8) << ";"
        << formatearDecimal(secuencial.tiempo_us, 8) << "\n";

    // Escribir datos paralelos
    archivo << "Paralelo;" << paralelo.samples << ";" << paralelo.num_hilos << ";"
        << formatearDecimal(paralelo.pi, 12) << ";"
        << formatearDecimal(paralelo.tiempo_segundos, 12) << ";"
        << formatearDecimal(paralelo.tiempo_ms, 8) << ";"
        << formatearDecimal(paralelo.tiempo_us, 8) << "\n";

    archivo.close();
    printf("Resultados guardados en: %s\n", nombre_archivo);
}



// Función principal que ejecuta ambos métodos
int main(int argc, char* argv[]) {
    long long samples = 3000;

    // Modificar las samples por consola si es necesario
    if (argc > 1)
        samples = atoll(argv[1]);

    // Ejecutamos la versión secuencial
    ResultadoMontecarlo resultado_secuencial = montecarlo_secuencial(samples);

    // Ejecutamos la versión paralela
    ResultadoMontecarlo resultado_paralelo = montecarlo_paralelo(samples);

    // Comparamos resultados
    printf("Comparacion de resultados:\n");
    printf("PI secuencial: %.12f\n", resultado_secuencial.pi);
    printf("PI paralelo:   %.12f\n", resultado_paralelo.pi);
    printf("Diferencia:    %.12f\n", fabs(resultado_secuencial.pi - resultado_paralelo.pi));
    
    // Guardar resultados en CSV
    guardar_csv(resultado_secuencial, resultado_paralelo, "resultados_montecarlo.csv");

    return 0;
}
