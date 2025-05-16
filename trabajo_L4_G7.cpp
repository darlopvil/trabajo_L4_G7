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

	a = omp_get_num_procs(); // Obtener el número de procesadores disponibles
    int num_threads = 4; // Número de hilos disponibles
    omp_set_num_threads(num_threads);
    resultado.num_hilos = num_threads;

    // Crear array de semillas con tamaño adecuado
    unsigned short* xi = new unsigned short[num_threads];
	srand(time(NULL)); // Inicializa el generador de números aleatorios con su semilla
    for (int j = 0; j < num_threads; j++) {
		xi[j] = rand() % 10000 + 1; // Genera semillas aleatorias entre 1 y 10000
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

// Función para guardar los resultados en un archivo CSV con soporte para múltiples pruebas
void guardar_csv(const ResultadoMontecarlo& secuencial, const ResultadoMontecarlo& paralelo,
    const char* nombre_archivo, bool primera_escritura = true) {
    // Modo de apertura: si es primera escritura, crea nuevo archivo, sino añade
    std::ofstream archivo;

    if (primera_escritura) {
        archivo.open(nombre_archivo); // Sobreescribe el archivo
    }
    else {
        archivo.open(nombre_archivo, std::ios::app); // Añade al archivo existente
    }

    if (!archivo.is_open()) {
        printf("Error: No se pudo abrir el archivo %s para escritura\n", nombre_archivo);
        return;
    }

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

    // Escribir cabecera solo si es la primera escritura
    if (primera_escritura) {
        // Usar punto y coma como separador y coma como decimal
        archivo << "Samples;Método;Hilos;Valor Pi;Tiempo (s);Tiempo (ms);Tiempo (us)\n";
    }

    // Escribir datos secuenciales
    archivo << secuencial.samples << ";Secuencial;" << secuencial.num_hilos << ";"
        << formatearDecimal(secuencial.pi, 12) << ";"
        << formatearDecimal(secuencial.tiempo_segundos, 12) << ";"
        << formatearDecimal(secuencial.tiempo_ms, 8) << ";"
        << formatearDecimal(secuencial.tiempo_us, 8) << "\n";

    // Escribir datos paralelos
    archivo << paralelo.samples << ";Paralelo;" << paralelo.num_hilos << ";"
        << formatearDecimal(paralelo.pi, 12) << ";"
        << formatearDecimal(paralelo.tiempo_segundos, 12) << ";"
        << formatearDecimal(paralelo.tiempo_ms, 8) << ";"
        << formatearDecimal(paralelo.tiempo_us, 8) << "\n";

    archivo.close();
}


// Función principal que ejecuta ambos métodos
int main(int argc, char* argv[]) {
    // Array con los tamaños de muestra a probar
    long long tamanos_muestra[] = { 3000, 300000, 3000000 };
    int num_pruebas = sizeof(tamanos_muestra) / sizeof(tamanos_muestra[0]);

    // Si se pasa un argumento por consola, solo ejecutar con ese tamaño
    if (argc > 1) {
        tamanos_muestra[0] = atoll(argv[1]);
        num_pruebas = 1;
    }

    const char* nombre_archivo = "resultados_montecarlo_todos.csv";
    printf("\n====== INICIANDO PRUEBAS CON DIFERENTES TAMANYOS DE MUESTRA ======\n\n");

    for (int i = 0; i < num_pruebas; i++) {
        long long samples = tamanos_muestra[i];
        printf("\n\n======= PRUEBA CON %lld MUESTRAS =======\n\n", samples);

        // Ejecutar versiones secuencial y paralela
        ResultadoMontecarlo resultado_secuencial = montecarlo_secuencial(samples);
        ResultadoMontecarlo resultado_paralelo = montecarlo_paralelo(samples);


        // Comparar resultados
        printf("Comparacion de resultados:\n");
        printf("PI secuencial: %.12f\n", resultado_secuencial.pi);
        printf("PI paralelo:   %.12f\n", resultado_paralelo.pi);
        printf("Diferencia:    %.12f\n", fabs(resultado_secuencial.pi - resultado_paralelo.pi));

        // Primera iteración crea nuevo archivo, las siguientes añaden
        guardar_csv(resultado_secuencial, resultado_paralelo, nombre_archivo, i == 0);
    }

    printf("\nTodos los resultados guardados en: %s\n", nombre_archivo);
    printf("\n====== TODAS LAS PRUEBAS COMPLETADAS ======\n");

    return 0;
}

