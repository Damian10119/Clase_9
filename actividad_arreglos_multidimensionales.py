"""
Actividad 5 - Arreglos y tipos abstractos de datos (Parte 2)
Solución de cinco problemas mediante arreglos bidimensionales y tridimensionales.


NOMBRE: Damian Pillajo
FECHA: 08/11/2026
CLASE: 8

"""

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt


OUTPUT_DIR = Path("resultados")
OUTPUT_DIR.mkdir(exist_ok=True)


# _________________________________________________________
#
# PROBLEMA 1: Análisis de fuerzas en una estructura 3x3
# _________________________________________________________

def problema_1_fuerzas():
    print("\nPROBLEMA 1: ANÁLISIS DE FUERZAS")

    # Cada posición contiene una carga vertical aplicada en el nodo.
    # Se consideran positivas las magnitudes de las cargas hacia abajo.
    cargas = np.array([
        [10.0,  0.0, 15.0],
        [ 0.0, 20.0,  0.0],
        [ 5.0,  0.0, 10.0]
    ])

    # Coordenadas de los tres apoyos: (x, y), donde x=columna e y=fila.
    apoyos = [(0.0, 0.0), (2.0, 0.0), (1.0, 2.0)]

    # Se calculan la carga total y sus momentos respecto a los ejes.
    carga_total = np.sum(cargas)
    momento_x_cargas = 0.0  # Sumatoria de carga * coordenada x
    momento_y_cargas = 0.0  # Sumatoria de carga * coordenada y

    for fila in range(3):
        for columna in range(3):
            fuerza = cargas[fila, columna]
            momento_x_cargas += fuerza * columna
            momento_y_cargas += fuerza * fila

    # Equilibrio: sum(R)=sum(cargas), sum(x*R)=sum(x*carga),
    # sum(y*R)=sum(y*carga). Las incógnitas son las reacciones verticales.
    matriz_equilibrio = np.array([
        [1.0, 1.0, 1.0],
        [p[0] for p in apoyos],
        [p[1] for p in apoyos]
    ])
    vector_cargas = np.array([
        carga_total, momento_x_cargas, momento_y_cargas
    ])

    reacciones = np.linalg.solve(matriz_equilibrio, vector_cargas)

    print("Matriz de cargas (unidades de fuerza):")
    print(cargas)
    print("Reacciones verticales en los apoyos:")
    for indice, reaccion in enumerate(reacciones, start=1):
        print(f"  Apoyo {indice} {apoyos[indice - 1]}: {reaccion:.2f}")

    # Comprobación de equilibrio vertical.
    print(f"Comprobación: cargas = {carga_total:.2f}, "
          f"reacciones = {np.sum(reacciones):.2f}")


# _____________________________________________________________________
#
# PROBLEMA 2: Simulación de presión en una cuadrícula tridimensional
# _____________________________________________________________________

def problema_2_fluido():
    print("\nPROBLEMA 2: SIMULACIÓN DE FLUIDO 3D")

    # La última dimensión almacena [presión, temperatura, vx, vy, vz].
    # Por tanto, el arreglo tiene forma (3, 3, 3, 5).
    fluido = np.zeros((3, 3, 3, 5), dtype=float)
    fluido[..., 0] = 100.0  # Presión inicial uniforme.
    fluido[..., 1] = 20.0   # Temperatura inicial en grados Celsius.

    # Se introduce un pulso de presión en la celda central.
    fluido[1, 1, 1, 0] = 150.0

    def vecinos_promedio(matriz, i, j, k):
        """Calcula el promedio de los vecinos válidos de una celda."""
        valores = []
        for di, dj, dk in [
            (-1, 0, 0), (1, 0, 0),
            (0, -1, 0), (0, 1, 0),
            (0, 0, -1), (0, 0, 1)
        ]:
            ni, nj, nk = i + di, j + dj, k + dk
            if 0 <= ni < 3 and 0 <= nj < 3 and 0 <= nk < 3:
                valores.append(matriz[ni, nj, nk])
        return np.mean(valores) if valores else matriz[i, j, k]

    # Se usa una copia para que todas las celdas se actualicen
    # simultáneamente a partir del estado del paso anterior.
    presion_anterior = fluido[..., 0].copy()
    presion_nueva = presion_anterior.copy()

    for i in range(3):
        for j in range(3):
            for k in range(3):
                promedio = vecinos_promedio(presion_anterior, i, j, k)
                # Modelo didáctico de difusión: la presión se aproxima
                # parcialmente al promedio de sus vecinos.
                presion_nueva[i, j, k] = (
                    0.6 * presion_anterior[i, j, k] + 0.4 * promedio
                )

    fluido[..., 0] = presion_nueva

    print("Presión inicial:")
    print(presion_anterior)
    print("Presión después de un paso de simulación:")
    print(fluido[..., 0])
    print("Estado almacenado por celda: presión, temperatura y velocidad "
          "(vx, vy, vz).")
    print("Nota: es un modelo simplificado de difusión, no un solver "
          "completo de dinámica de fluidos.")


# _____________________________________________________________________
#
# PROBLEMA 3: Suavizado de un volumen de imágenes médicas 3D
# _____________________________________________________________________
def problema_3_imagenes():
    print("\nPROBLEMA 3: SUAVIZADO DE IMÁGENES MÉDICAS 3D")

    # Volumen sintético 3x5x5: tres capas 2D con intensidades de ejemplo.
    # En un caso real, este arreglo contendría los vóxeles de una imagen.
    volumen = np.array([
        [[10, 12, 11, 13, 10],
         [11, 80, 12, 14, 11],
         [10, 12, 13, 12, 10],
         [12, 11, 10, 12, 11],
         [10, 11, 12, 10, 10]],

        [[11, 12, 13, 12, 11],
         [12, 14, 90, 13, 12],
         [11, 13, 14, 12, 11],
         [10, 12, 13, 11, 10],
         [11, 10, 12, 13, 11]],

        [[10, 11, 12, 10, 10],
         [12, 13, 12, 11, 12],
         [11, 12, 75, 13, 11],
         [10, 11, 12, 12, 10],
         [11, 10, 11, 12, 10]]
    ], dtype=float)

    def promedio_3x3(capa, fila, columna):
        """Promedia los píxeles vecinos dentro de una ventana 3x3."""
        fila_inicio = max(0, fila - 1)
        fila_fin = min(capa.shape[0], fila + 2)
        col_inicio = max(0, columna - 1)
        col_fin = min(capa.shape[1], columna + 2)
        ventana = capa[fila_inicio:fila_fin, col_inicio:col_fin]
        return np.mean(ventana)

    volumen_suavizado = np.empty_like(volumen)

    # Se aplica el filtro independientemente a cada capa 2D.
    for profundidad in range(volumen.shape[0]):
        for fila in range(volumen.shape[1]):
            for columna in range(volumen.shape[2]):
                volumen_suavizado[profundidad, fila, columna] = (
                    promedio_3x3(volumen[profundidad], fila, columna)
                )

    print("Primera capa original:")
    print(volumen[0])
    print("Primera capa suavizada:")
    print(np.round(volumen_suavizado[0], 2))

    # Visualización de la primera capa antes y después del filtro.
    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    imagen_original = axes[0].imshow(volumen[0], cmap="gray")
    axes[0].set_title("Capa original")
    axes[1].imshow(volumen_suavizado[0], cmap="gray")
    axes[1].set_title("Capa suavizada")
    for ax in axes:
        ax.set_xlabel("Columna")
        ax.set_ylabel("Fila")
    fig.colorbar(imagen_original, ax=axes, label="Intensidad")
    fig.suptitle("Filtro de promedio 3x3 - primera capa")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "problema_3_suavizado.png", dpi=150)
    plt.close(fig)


# _____________________________________________________________________
#
# PROBLEMA 4: Análisis de una tabla de sensores 5x5
# _____________________________________________________________________

def problema_4_sensores():
    print("\nPROBLEMA 4: TABLA DE SENSORES")

    # Filas = momentos de medición; columnas = sensores.
    temperaturas = np.array([
        [21.2, 22.1, 20.8, 21.5, 22.0],
        [21.5, 22.4, 21.0, 21.8, 22.2],
        [22.0, 22.7, 21.4, 22.1, 22.5],
        [22.3, 23.0, 21.7, 22.4, 22.8],
        [22.1, 22.8, 21.5, 22.2, 22.6]
    ])

    # axis=0 calcula estadísticas por columna (cada sensor).
    promedio_sensor = np.mean(temperaturas, axis=0)
    desviacion_sensor = np.std(temperaturas, axis=0)

    # axis=1 calcula estadísticas por fila (cada momento).
    promedio_momento = np.mean(temperaturas, axis=1)
    desviacion_momento = np.std(temperaturas, axis=1)

    print("Tabla de temperaturas:")
    print(temperaturas)
    print("Promedio por sensor:", np.round(promedio_sensor, 2))
    print("Desviación estándar por sensor:",
          np.round(desviacion_sensor, 2))
    print("Promedio por momento:", np.round(promedio_momento, 2))
    print("Desviación estándar por momento:",
          np.round(desviacion_momento, 2))

    # Gráfica de temperaturas de los cinco sensores a través del tiempo.
    fig, ax = plt.subplots(figsize=(8, 4.5))
    momentos = np.arange(1, 6)
    for sensor in range(5):
        ax.plot(momentos, temperaturas[:, sensor], marker="o",
                label=f"Sensor {sensor + 1}")
    ax.set_title("Lecturas de temperatura por sensor")
    ax.set_xlabel("Momento de medición")
    ax.set_ylabel("Temperatura (°C)")
    ax.set_xticks(momentos)
    ax.grid(True)
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "problema_4_sensores.png", dpi=150)
    plt.close(fig)


# ________________________________________________________
#
# PROBLEMA 5: Transformación lineal de coordenadas
# ________________________________________________________

def problema_5_coordenadas():
    print("\nPROBLEMA 5: TRANSFORMACIÓN DE COORDENADAS")

    # Cada fila representa un punto [x, y].
    puntos = np.array([
        [1.0, 1.0],
        [3.0, 1.0],
        [2.0, 3.0],
        [-1.0, 2.0]
    ])

    # Matriz de rotación de 45 grados en sentido antihorario.
    angulo = np.radians(45)
    matriz_rotacion = np.array([
        [np.cos(angulo), -np.sin(angulo)],
        [np.sin(angulo),  np.cos(angulo)]
    ])

    # Se usa la multiplicación matricial para transformar todos los puntos.
    # Los puntos están organizados por filas, por eso se multiplica por R.T.
    puntos_transformados = puntos @ matriz_rotacion.T

    print("Puntos originales:")
    print(puntos)
    print("Puntos después de rotar 45 grados:")
    print(np.round(puntos_transformados, 2))

    # Representación gráfica antes y después de la rotación.
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(puntos[:, 0], puntos[:, 1], label="Original", marker="o")
    ax.scatter(puntos_transformados[:, 0], puntos_transformados[:, 1],
               label="Rotados", marker="x")
    for indice in range(len(puntos)):
        ax.plot(
            [puntos[indice, 0], puntos_transformados[indice, 0]],
            [puntos[indice, 1], puntos_transformados[indice, 1]],
            linestyle="--", alpha=0.5
        )
    ax.axhline(0, linewidth=0.8)
    ax.axvline(0, linewidth=0.8)
    ax.set_aspect("equal", adjustable="box")
    ax.set_title("Rotación de puntos cartesianos (45°)")
    ax.set_xlabel("Coordenada X")
    ax.set_ylabel("Coordenada Y")
    ax.grid(True)
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "problema_5_coordenadas.png", dpi=150)
    plt.close(fig)


def main():
    """Ejecuta los cinco ejercicios en orden."""
    problema_1_fuerzas()
    problema_2_fluido()
    problema_3_imagenes()
    problema_4_sensores()
    problema_5_coordenadas()

    print("\nProceso finalizado.")
    print(f"Las gráficas se guardaron en: {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
