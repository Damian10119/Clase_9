# Actividad 5: Arreglos y tipos abstractos de datos (Parte 2)

## Descripción

Este proyecto resuelve cinco problemas prácticos de ingeniería utilizando arreglos bidimensionales y tridimensionales en Python. Se emplea NumPy para almacenar y procesar datos numéricos y Matplotlib para generar las visualizaciones.


## Soluciones y enfoque utilizado

### Problema 1. Análisis de fuerzas en una estructura

Se representa la estructura mediante una matriz de 3x3. Cada elemento almacena la magnitud de una carga vertical aplicada en un nodo. Se establecen tres apoyos y se plantea un sistema de tres ecuaciones de equilibrio: suma de fuerzas verticales y suma de momentos respecto a los ejes X e Y. NumPy resuelve el sistema lineal mediante `numpy.linalg.solve()`.

**Eficiencia:** la matriz permite acceder directamente a la carga de cada nodo. La formulación matricial evita resolver las ecuaciones manualmente y permite extender el cálculo a otros valores de carga y coordenadas. El ejemplo supone una estructura idealizada con tres reacciones verticales independientes; no sustituye un análisis estructural completo.

### Problema 2. Simulación de fluido en una cuadrícula 3D

Se utiliza un arreglo de dimensiones `3x3x3x5`. Las tres primeras dimensiones representan la posición espacial y la última almacena cinco variables por celda: presión, temperatura y las tres componentes de velocidad (`vx`, `vy`, `vz`). Se introduce un pulso de presión en el centro y se actualiza la presión de cada celda usando el promedio de sus vecinos inmediatos.

**Eficiencia:** el arreglo multidimensional organiza todas las variables del volumen en una estructura uniforme. Se utiliza una copia del estado anterior para que las actualizaciones de un paso no alteren los cálculos de las demás celdas. La regla aplicada es un modelo didáctico de difusión de presión, no una simulación física completa de dinámica de fluidos.

### Problema 3. Análisis y suavizado de imágenes médicas 3D

Se representa un volumen mediante un arreglo tridimensional: profundidad, filas y columnas. Cada capa es una imagen 2D. Para reducir variaciones locales, se aplica a cada píxel un filtro de promedio de ventana 3x3, considerando únicamente los píxeles que están dentro de los límites de la capa. Se muestra la primera capa antes y después del procesamiento.

**Eficiencia:** el arreglo 3D permite recorrer y procesar las capas de manera ordenada. El filtro utiliza ventanas pequeñas y operaciones locales, por lo que es sencillo de implementar y comprender. El promedio puede reducir ruido, aunque también puede suavizar bordes y detalles importantes; para uso clínico se requieren métodos validados y datos médicos reales.

### Problema 4. Manejo de datos en una tabla de sensores

Se crea una matriz de 5x5: cada fila corresponde a un momento de medición y cada columna a un sensor. Con `numpy.mean()` y `numpy.std()` se calculan promedios y desviaciones estándar por sensor (`axis=0`) y por momento (`axis=1`). Finalmente, se grafica la evolución de las cinco lecturas.

**Eficiencia:** NumPy realiza los cálculos sobre filas y columnas sin necesitar ciclos manuales para cada promedio. Esto reduce la cantidad de código y aprovecha operaciones numéricas optimizadas. La desviación estándar utilizada es la poblacional (`ddof=0`).

### Problema 5. Transformación de coordenadas cartesianas

Los puntos se almacenan en una matriz donde cada fila contiene las coordenadas `[x, y]`. Se construye una matriz de rotación de 45 grados y se aplica a todos los puntos mediante multiplicación matricial.

**Eficiencia:** una sola operación matricial transforma todos los puntos, en lugar de calcular cada coordenada individualmente. El mismo enfoque puede adaptarse a otras rotaciones, escalados o transformaciones lineales.

## Archivos generados

Al ejecutar el programa se crea la carpeta `resultados` con:

- `problema_3_suavizado.png`: comparación de una capa antes y después del filtro.
- `problema_4_sensores.png`: evolución de las temperaturas de los sensores.
- `problema_5_coordenadas.png`: puntos originales y puntos rotados.

