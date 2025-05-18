/******************************************************************************
 * CÁLCULO DE PI MEDIANTE EL MÉTODO DE MONTE CARLO
 *****************************************************************************
 *
 * IMPLEMENTACIÓN:
 * --------------
 * 1. Generamos puntos aleatorios (x,y) donde x,y ∈ [0,1] (primer cuadrante)
 * 2. Comprobamos si el punto está dentro del círculo: x² + y² ≤ 1
 * 3. Calculamos π como: 4 * (puntos_dentro / total_puntos)
 *
 * PARALELIZACIÓN:
 * --------------
 * El cálculo de Monte Carlo es ideal para paralelizar porque:
 *   - Cada punto es independiente (no hay dependencias entre iteraciones)
 *   - Solo necesitamos agregar un contador global (operación de reducción)
 *   - La carga computacional se distribuye uniformemente
 *
 * DESAFÍOS EN LA GENERACIÓN ALEATORIA PARALELA:
 * --------------------------------------------
 * En entornos paralelos, es crítico que cada hilo tenga su propio generador
 * de números aleatorios para evitar:
 *   - Competencia por recursos compartidos
 *   - Secuencias idénticas que reducen la calidad estadística
 *   - Comportamiento no determinista
 *
 * En esta implementación:
 *   - Versión secuencial: Usa rand()/RAND_MAX (generador simple de C)
 *   - Versión paralela: Usa std::mt19937 (Mersenne Twister, alta calidad)
 *     con semillas únicas para cada hilo
 */

#include <stdio.h>
#include <math.h>
#include <omp.h>      // Biblioteca OpenMP para paralelización
#include <stdlib.h>
#include <time.h>
#include <fstream>    // Para manejo de archivos
#include <random>     // Para generadores aleatorios de alta calidad (C++11)

 // Estructura para almacenar los resultados de ambos métodos (secuencial y paralelo)
struct ResultadoMontecarlo {
	double pi;                // Valor calculado de π
	double tiempo_segundos;   // Tiempo de ejecución en segundos
	double tiempo_ms;         // Tiempo de ejecución en milisegundos
	double tiempo_us;         // Tiempo de ejecución en microsegundos
	long long samples;        // Número de muestras utilizadas
	bool es_paralelo;         // Indica si es versión paralela o secuencial
	int num_hilos;            // Número de hilos usados (1 para secuencial)
};

/**
 * IMPLEMENTACIÓN SECUENCIAL DEL MÉTODO DE MONTE CARLO
 *
 * @param samples: Número de puntos aleatorios a generar
 * @return ResultadoMontecarlo: Estructura con el valor de π y estadísticas de tiempo
 */
ResultadoMontecarlo montecarlo_secuencial(long long samples) {
	unsigned long long count = 0;  // Contador de puntos dentro del círculo
	unsigned long long i;
	double x, y;                   // Coordenadas del punto aleatorio
	double inicio, final, total = 0;
	ResultadoMontecarlo resultado;

	// Inicializar datos del resultado
	resultado.samples = samples;
	resultado.es_paralelo = false;
	resultado.num_hilos = 1;

	// Iniciar cronómetro
	inicio = omp_get_wtime();

	// Bucle principal - genera 'samples' puntos aleatorios
	for (i = 0; i < samples; ++i) {
		// Generar punto aleatorio en el cuadrante [0,1]×[0,1]
		// rand() genera enteros entre 0 y RAND_MAX
		// La división normaliza a valores entre 0 y 1
		x = ((double)rand()) / ((double)RAND_MAX);
		y = ((double)rand()) / ((double)RAND_MAX);

		// Comprobar si el punto está dentro del círculo unitario
		// Un punto (x,y) está dentro del círculo si x² + y² ≤ 1
		if (x * x + y * y <= 1.0) {
			++count;  // Incrementar contador si está dentro
		}
	}

	// Detener cronómetro y calcular tiempo total
	final = omp_get_wtime();
	total = (final - inicio);

	// Calcular π: 4 veces la proporción de puntos dentro del círculo
	// Multiplicamos por 4 porque solo estamos considerando un cuadrante
	resultado.pi = 4.0 * count / samples;
	resultado.tiempo_segundos = total;
	resultado.tiempo_ms = total * 1e3;  // convertir a milisegundos
	resultado.tiempo_us = total * 1e6;  // convertir a microsegundos

	// Mostrar resultados por consola
	printf("----------------OpenMP MonterCarlo Sin Paralelizar----------------\n");
	printf("Numero de Samples = %lld\n", samples);
	printf("pi = %.12f\n", resultado.pi);
	printf("Tiempo de ejec./elemento de calculo (en segundos) => %.12lf s\n", resultado.tiempo_segundos);
	printf("Tiempo de ejec./elemento de calculo (en milisegundos) => %.8lf ms\n", resultado.tiempo_ms);
	printf("Tiempo de ejec./elemento de calculo (en microsegundos) => %.8lf us\n", resultado.tiempo_us);
	printf("-------------------------------------------------------------------\n\n");

	return resultado;
}

/**
 * IMPLEMENTACIÓN PARALELA DEL MÉTODO DE MONTE CARLO USANDO OPENMP
 *
 * @param samples: Número de puntos aleatorios a generar
 * @return ResultadoMontecarlo: Estructura con el valor de π y estadísticas de tiempo
 */
ResultadoMontecarlo montecarlo_paralelo(long long samples) {
	unsigned long long count = 0;  // Contador global (compartido entre hilos)
	long long i, a;
	double x, y;                   // Variables para coordenadas (privadas por hilo)
	double inicio, final, total = 0;
	ResultadoMontecarlo resultado;

	// Inicializar datos del resultado
	resultado.samples = samples;
	resultado.es_paralelo = true;

	// Configuración de paralelismo
	a = omp_get_num_procs();       // Obtener número de procesadores físicos
	int num_threads = 8;           // Establecer número de hilos
	omp_set_num_threads(num_threads);
	resultado.num_hilos = num_threads;

	// Iniciar cronómetro
	inicio = omp_get_wtime();

	// Inicialización del generador de semillas de alta calidad
	std::random_device rd;         // Obtiene entropía del hardware si está disponible
	unsigned int seed_base = rd(); // Semilla base compartida entre todos los hilos

	// Inicio de la región paralela
#pragma omp parallel private(x,y)
	{
		// Código ejecutado por cada hilo:

		// 1. Obtener ID único del hilo actual
		int tid = omp_get_thread_num();

		// 2. Crear semilla única para este hilo
		// Usamos XOR con una constante derivada de la proporción áurea (0x9e3779b9)
		// para dispersar bien los valores y evitar correlaciones
		unsigned int seed = seed_base ^ (static_cast<unsigned int>(tid) + 1) * 0x9e3779b9;

		// 3. Inicializar generador Mersenne Twister con esta semilla
		// Este generador tiene excelentes propiedades estadísticas:
		// - Período extremadamente largo (2^19937-1)
		// - Distribución uniforme en 623 dimensiones
		// - Pasa todos los tests estadísticos conocidos
		std::mt19937 gen(seed);

		// 4. Configurar distribución uniforme real en [0,1)
		std::uniform_real_distribution<double> dis(0.0, 1.0);

		// 5. Repartir iteraciones entre los hilos disponibles
		// La cláusula reduction(+:count) combina automáticamente los contadores parciales
#pragma omp for reduction(+:count)
		for (i = 0; i < samples; ++i) {
			// Generar par de coordenadas aleatorias usando nuestro generador de alta calidad
			x = dis(gen);
			y = dis(gen);

			// Comprobar si el punto está dentro del círculo unitario
			if (x * x + y * y <= 1.0) {
				++count;  // Incrementar contador si está dentro
			}
		}
		// Al final del bloque paralelo, OpenMP combina automáticamente todos los
		// contadores parciales en la variable count (gracias a reduction)
	}

	// Detener cronómetro y calcular tiempo
	final = omp_get_wtime();
	total = (final - inicio);

	// Calcular π y almacenar resultados
	resultado.pi = 4.0 * count / samples;
	resultado.tiempo_segundos = total;
	resultado.tiempo_ms = total * 1e3;
	resultado.tiempo_us = total * 1e6;

	// Mostrar resultados por consola
	printf("----------------OpenMP MonterCarlo Paralelizado----------------\n");
	printf("Numero de Procesadores: %lld\n", a);
	printf("Numero de Hilos utilizados: %d\n", num_threads);
	printf("Numero de Samples = %lld\n", samples);
	printf("pi = %.12f\n", resultado.pi);
	printf("Tiempo de ejec./elemento de calculo (en segundos) => %.12lf s\n", resultado.tiempo_segundos);
	printf("Tiempo de ejec./elemento de calculo (en milisegundos) => %.8lf ms\n", resultado.tiempo_ms);
	printf("Tiempo de ejec./elemento de calculo (en microsegundos) => %.8lf us\n", resultado.tiempo_us);
	printf("-------------------------------------------------------------------\n\n");

	return resultado;
}

/**
 * Función para guardar los resultados en un archivo CSV con formato español
 *
 * @param secuencial: Resultados del método secuencial
 * @param paralelo: Resultados del método paralelo
 * @param nombre_archivo: Ruta del archivo CSV a crear/modificar
 * @param primera_escritura: Si es true, crea nuevo archivo; si es false, añade al existente
 */
void guardar_csv(const ResultadoMontecarlo& secuencial, const ResultadoMontecarlo& paralelo,
	const char* nombre_archivo, bool primera_escritura = true) {
	// Abrir el archivo en modo apropiado
	std::ofstream archivo;

	if (primera_escritura) {
		archivo.open(nombre_archivo);              // Modo sobrescritura
	}
	else {
		archivo.open(nombre_archivo, std::ios::app); // Modo append (añadir)
	}

	// Verificar que el archivo se abrió correctamente
	if (!archivo.is_open()) {
		printf("Error: No se pudo abrir el archivo %s para escritura\n", nombre_archivo);
		return;
	}

	// Función lambda para formatear números con coma decimal (formato español)
	auto formatearDecimal = [](double valor, int precision) -> std::string {
		char buffer[64];
		// Formatear con precisión específica
		snprintf(buffer, sizeof(buffer), "%.*f", precision, valor);
		std::string resultado = buffer;
		// Convertir punto a coma para Excel español
		for (char& c : resultado) {
			if (c == '.') c = ',';
		}
		return resultado;
		};

	// Escribir encabezados solo en la primera escritura
	if (primera_escritura) {
		// Usar punto y coma como separador de campos (CSV español)
		archivo << "Samples;Método;Hilos;Valor Pi;Tiempo (s);Tiempo (ms);Tiempo (us)\n";
	}

	// Escribir resultados del método secuencial
	archivo << secuencial.samples << ";Secuencial;" << secuencial.num_hilos << ";"
		<< formatearDecimal(secuencial.pi, 12) << ";"
		<< formatearDecimal(secuencial.tiempo_segundos, 12) << ";"
		<< formatearDecimal(secuencial.tiempo_ms, 8) << ";"
		<< formatearDecimal(secuencial.tiempo_us, 8) << "\n";

	// Escribir resultados del método paralelo
	archivo << paralelo.samples << ";Paralelo;" << paralelo.num_hilos << ";"
		<< formatearDecimal(paralelo.pi, 12) << ";"
		<< formatearDecimal(paralelo.tiempo_segundos, 12) << ";"
		<< formatearDecimal(paralelo.tiempo_ms, 8) << ";"
		<< formatearDecimal(paralelo.tiempo_us, 8) << "\n";

	// Cerrar el archivo
	archivo.close();
}

/**
 * Función principal del programa
 *
 * Ejecuta ambos métodos (secuencial y paralelo) con diferentes tamaños de muestra,
 * comparando resultados y tiempos de ejecución.
 */
int main(int argc, char* argv[]) {
	// Definir 10 tamaños de muestra para las pruebas, desde miles hasta casi 100 millones
	long long tamanos_muestra[] = {
		1000,           // 1 mil - evaluación muy rápida
		5000,           // 5 mil
		10000,          // 10 mil - evaluación rápida
		50000,          // 50 mil
		100000,         // 100 mil
		500000,         // 500 mil
		1000000,        // 1 millón - buena precisión
		5000000,        // 5 millones
		10000000,       // 10 millones - alta precisión
		50000000        // 50 millones - muy alta precisión
	};

	int num_pruebas = sizeof(tamanos_muestra) / sizeof(tamanos_muestra[0]);

	// Procesar argumentos de línea de comandos si existen
	if (argc > 1) {
		// Si el usuario proporciona un tamaño, usar solo ese
		tamanos_muestra[0] = atoll(argv[1]);
		num_pruebas = 1;
	}

	// Nombre del archivo CSV para guardar resultados
	const char* nombre_archivo = "resultados_montecarlo_openmp.csv";
	printf("\n====== INICIANDO PRUEBAS CON DIFERENTES TAMANYOS DE MUESTRA ======\n\n");

	// Ejecutar pruebas para cada tamaño de muestra
	for (int i = 0; i < num_pruebas; i++) {
		long long samples = tamanos_muestra[i];
		printf("\n\n======= PRUEBA CON %lld MUESTRAS =======\n\n", samples);

		// Ejecutar ambas versiones
		ResultadoMontecarlo resultado_secuencial = montecarlo_secuencial(samples);
		ResultadoMontecarlo resultado_paralelo = montecarlo_paralelo(samples);

		// Comparar precisión de los resultados
		printf("Comparacion de resultados:\n");
		printf("PI secuencial: %.12f\n", resultado_secuencial.pi);
		printf("PI paralelo:   %.12f\n", resultado_paralelo.pi);
		printf("Diferencia:    %.12f\n", fabs(resultado_secuencial.pi - resultado_paralelo.pi));

		// Guardar resultados en CSV (primera iteración crea archivo, las siguientes añaden)
		guardar_csv(resultado_secuencial, resultado_paralelo, nombre_archivo, i == 0);
	}

	printf("\nTodos los resultados guardados en: %s\n", nombre_archivo);
	printf("\n====== TODAS LAS PRUEBAS COMPLETADAS ======\n");

	return 0;
}
