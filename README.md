# Volatilidad de Bitcoin a 10 segundos con Fourier

A partir de cierres de BTC a un segundo, el notebook construye barras de diez segundos y estudia tres preguntas. ¿Cuánta historia hace falta para representar el comportamiento del precio y de los retornos? ¿Cómo evoluciona la volatilidad dentro de una ventana? ¿Qué precisión tiene la última estimación disponible al anticipar la variación del siguiente intervalo?

**Para empezar**, abrí el notebook principal y ejecutá las dos primeras celdas de código. Si te interesa la estimación espectral, podés saltar desde ahí a la sección **Fourier · volatilidad spot de BTC a 10 segundos**. El repositorio también contiene un segundo cuaderno, [Fourier_BTC.ipynb](./Fourier_BTC.ipynb). Las instrucciones y cifras de esta guía corresponden a [Fourier_10seg_2325.ipynb](./Fourier_10seg_2325.ipynb).

[Abrir la versión de GitHub en Colab](https://colab.research.google.com/github/Miguithub/fourier/blob/main/Fourier_10seg_2325.ipynb) · [Abrir el Colab original en la celda compartida](https://colab.research.google.com/drive/1TAfrU9R3j_FacX8MpO7MUiKJu4Jfmmbz#scrollTo=btc-eda-003)

## Datos de entrada

El insumo es `BTC_1s_2023_2025.parquet`, un archivo externo al repositorio. Se necesitan dos campos: una marca temporal por observación y el precio de cierre `close`. La preparación inspecciona el esquema antes de cargar datos y proyecta únicamente las columnas necesarias.

| Ajuste inicial | Valor incluido | Cuándo cambiarlo |
| --- | --- | --- |
| `BTC_PARQUET_PATH` | Variable de entorno opcional | Para indicar la ruta completa a una copia local o montada. Tiene prioridad. |
| `TIMESTAMP_COLUMN` | `None` | Indicá el nombre exacto si hay más de un candidato temporal. |
| `TIMESTAMP_UNIT` | `None` | Para una época numérica, especificá `s`, `ms`, `us` o `ns` según el origen. |
| `SOURCE_TIMEZONE` | `UTC` | Para fechas sin zona, usá la zona que corresponde al registro original. |
| `MEMORY_LIMIT`, `THREADS` | `1GB`, `2` | Controlan DuckDB. NumPy, pandas y el entrenamiento también consumen memoria. |
| `RUN_R_BAI_PERRON` | `False` | Activá esta opción si querés ejecutar la variante en R y disponés de `Rscript` y `strucchange`. |

### Ejecución en Google Colab

1. Abrí el enlace de Colab de arriba. Para trabajar con tu propia copia del Parquet, definí su ruta en una celda previa mediante `%env BTC_PARQUET_PATH=/content/datos/BTC_1s_2023_2025.parquet`.
2. Si faltan dependencias, ejecutá `%pip install duckdb pyarrow numpy pandas scipy statsmodels matplotlib`. La instalación aparece como comentario en el notebook. Para entrenar la LSTM también se necesita TensorFlow.
3. Ejecutá la primera celda de código. Si no definiste una ruta, intenta montar Drive, busca el archivo en `MyDrive/Neural/BTC/LSTM+`, revisa el directorio actual y, como último recurso, descarga el Parquet por su ID con la cuenta autenticada. Este último camino requiere acceso de descarga al archivo y las bibliotecas `google-api-python-client` y `google-auth`.
4. Comprobá las líneas **Fuente localizada** y **Resultados**. La descarga por ID comprueba el tamaño y, cuando está disponible, el MD5. Reserva el tamaño del Parquet más 2 GiB libres para esta etapa. La ordenación y las salidas requieren espacio adicional.
5. Ejecutá la segunda celda de código y revisá los conteos de calidad. Cuando termina, queda disponible `data10` para las secciones siguientes. Si la salida apunta al disco temporal de Colab, copiá los resultados que quieras conservar antes de cerrar la sesión.

### Ejecución local con Jupyter

Desde la raíz del repositorio, con un entorno Python activo, un ejemplo en PowerShell es:

```powershell
python -m pip install jupyterlab ipykernel duckdb pyarrow numpy pandas scipy statsmodels matplotlib
$env:BTC_PARQUET_PATH = 'D:\datos\BTC_1s_2023_2025.parquet'
python -m jupyterlab Fourier_10seg_2325.ipynb
```

Reemplazá la ruta por la de tu archivo. Elegí el kernel del entorno donde instalaste las dependencias. La celda de acceso y la de preparación deben ejecutarse en el mismo kernel que las secciones analíticas. Para la extensión neuronal, instalá TensorFlow en ese entorno antes de llamar a `train_tail`.

## De observaciones de un segundo a variables de diez segundos

La preparación ordena por tiempo y comprueba duplicados, marcas inválidas o fuera de la rejilla de segundos y cierres inválidos o no positivos. Los huecos interrumpen los retornos. El primer bloque puede quedar incompleto porque falta el cierre anterior al primer segundo observado.

| Serie | Cálculo por barra válida | Interpretación |
| --- | --- | --- |
| `close` | Último cierre del bloque | Nivel del precio, sensible a cambios de escala entre períodos. |
| `close_ln` | Logaritmo natural de `close` | Nivel en escala logarítmica. Puede conservar tendencia. |
| `diff_close_ln` | Diferencia entre logcierres consecutivos a 10 s | Retorno con signo que entra en la reconstrucción de Fourier. |
| `r2` | Suma de los diez cuadrados de retornos logarítmicos de 1 s | Variación realizada del intervalo. Su raíz está en unidades de retorno por barra. |

En particular, dos trayectorias con los mismos cierres inicial y final pueden tener idéntico `diff_close_ln` y distinto `r2`: la suma de cuadrados conserva el movimiento dentro de los diez segundos.

## Recorridos de análisis

### 1. Representatividad y backtest temporal

Ejecutá en orden las definiciones de funciones, quiebres, distancias y backtest, luego las cuatro llamadas a `analyze_variable` y la síntesis conjunta. Cada variable tiene sus propios gráficos y tablas. Las envolventes temporales resumen rangos y medias; los histogramas incorporan las observaciones válidas. ACF y PACF describen dependencia por rezago. ADF y KPSS contrastan hipótesis distintas sobre estacionariedad. Las pruebas de Chow toman como fechas prefijadas el 1 de enero de 2024 y el 1 de enero de 2025.

La rejilla `GRID_N` abarca de 60 a 777.600 barras (10 minutos a 90 días). Para comparar la distribución de una ventana con el historial, el cuaderno comparte bordes de histogramas, suaviza los conteos y calcula divergencias KL en ambas direcciones. También analiza la sensibilidad al número de bins y las distancias entre cuantiles.

El backtest anticipa un paso de 10 s con AR(1) y AR(5). BIC selecciona el orden por ventana. Los orígenes de desarrollo están entre el 60 % y el 80 % de la línea temporal; el último 20 % se reserva para confirmación. La comparación incluye persistencia, cero para retornos y media de entrenamiento para `r2`. La selección predictiva usa desarrollo. La selección equilibrada incorpora además la semejanza con el historial completo, por lo que su interpretación es retrospectiva.

### 2. Volatilidad spot y pronóstico Fourier

Después de preparar `data10`, ejecutá las dos celdas de definiciones Fourier y luego `analyze_fourier(n)` para la duración deseada. Los bloques son consecutivos, disjuntos y están alineados en UTC. Aquí **n cuenta incrementos**. Hacen falta 61 precios para formar una ventana de 60 retornos.

| Duración | 5 min | 10 min | 12 min | 15 min | 20 min | 1 h | 6 h | 24 h | 65 h | 7 días |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| n | 30 | 60 | 72 | 90 | 120 | 360 | 2.160 | 8.640 | 23.400 | 60.480 |

Dentro de cada bloque, el algoritmo estima el ruido y usa pilotos de varianza integrada, cuarticidad y variación de la varianza para elegir los truncamientos espectrales **N** y **M** mediante el criterio c-AMISE. La inversión de Fejér reconstruye la varianza en los puntos interiores. Las estimaciones negativas quedan registradas y carecen de raíz real. El caso de 23.400 incrementos conserva el conteo de la aplicación empírica del artículo; en BTC a diez segundos cubre 65 horas. [Metodología de Mancino, Mariotti y Toscano](https://arxiv.org/pdf/2209.08967v1)

La trayectoria interior usa el bloque completo y está disponible al cierre. Para evaluar un **pronóstico**, el cuaderno toma la última estimación interior, la mantiene durante los diez segundos siguientes y la compara con el `r2` observado en ese intervalo. La referencia de persistencia usa el último `r2` conocido.

| Diagnóstico | Qué pregunta responde | Cómo leer la salida |
| --- | --- | --- |
| Mincer–Zarnowitz | ¿Son compatibles el intercepto con cero y la pendiente con uno? | El Wald conjunto con covarianza HAC evalúa calibración. |
| Diebold–Mariano | ¿Difiere el error cuadrático medio de Fourier del de persistencia? | `delta_mse` positivo indica mayor error medio de Fourier. El valor p corresponde al contraste bilateral. |
| Ljung–Box | ¿Hay autocorrelación en los errores de Fourier? | Los rezagos se cuentan en bloques de la duración elegida. |

La prueba estándar usa el tramo regular más largo de pares válidos. Para 10 minutos, `mz_all_10min()` amplía el contraste de calibración a todos los pares finitos y pondera la covarianza por distancia real de reloj. Conviene leer siempre las columnas `elegibles` y `usados` junto al valor p.

### 3. LSTM para los últimos dos o tres minutos

Primero generá `n_60/ventanas.parquet` con la sección Fourier de 10 minutos. Ejecutá las definiciones de episodios y entrenamiento, fijá `N_LSTM = 60` y corré `build_samples(N_LSTM)`. El Parquet de episodios se reutiliza en las dos llamadas a `train_tail`.

| Llamada | Historia visible | Objetivo |
| --- | --- | --- |
| `train_tail(n=60, h=12)` | 48 pasos, 8 minutos | 12 pasos futuros, 2 minutos |
| `train_tail(n=60, h=18)` | 42 pasos, 7 minutos | 18 pasos futuros, 3 minutos |

Cada modelo recibe retornos en puntos básicos, amplitudes absolutas, coeficientes de Fourier y detalles Haar calculados hacia atrás. La LSTM de 32 unidades predice dos trayectorias, una etiqueta Fourier reconstruida con el episodio completo y una etiqueta Haar suavizada. Ambas comparten observaciones con el prefijo. Para interpretar las métricas, el cuaderno también compara los pronósticos con la raíz de la variación realizada en cada barra futura y con la persistencia disponible al corte. La partición cronológica es 70/15/15 y excluye episodios que crucen fronteras entre conjuntos.

## Resultados que ya se pueden inspeccionar

Las cifras provienen de las **salidas guardadas en el notebook enlazado**. La tabla de preparación registra **86.740.818 observaciones** desde el 1 de enero de 2023 a las 00:00:00 hasta el 30 de septiembre de 2025 a las 22:40:17 UTC. Reporta cero timestamps duplicados y **8.674.080 barras válidas** de diez segundos sobre 8.674.082 bloques observados.

- **Selección de historia.** Con objetivo 0,95, la síntesis devuelve `n_predictive = 60` para cierre, logcierre y retorno, y `360` para `r2`. Ningún candidato satisfizo todos los criterios de selección equilibrada. Esos tamaños describen el backtest AR a un paso.
- **Fourier con bloques de 10 minutos.** El contraste estándar usa 9.541 de 144.460 pares elegibles. Mincer–Zarnowitz rechaza calibración conjunta (Wald 335,61). Diebold–Mariano arroja p ≈ 0,408, de modo que esa muestra no permite distinguir el error cuadrático medio de las dos reglas al 5 %.
- **Fourier con bloques de 24 horas.** Los 1.002 pares elegibles entran al contraste. El diferencial de error cuadrático es positivo y Diebold–Mariano arroja p ≈ 0,0194, favorable a persistencia en ese ejercicio. El horizonte pronosticado sigue siendo diez segundos.
- **Extensiones.** Las celdas de 12 y 15 minutos y los entrenamientos LSTM carecen de salidas guardadas. Para obtener métricas neuronales por horizonte hay que generar los episodios y ejecutar ambos entrenamientos.

## Archivos generados y resolución de problemas

La primera celda imprime `OUT`. Usa la carpeta `EDA_close_10s_resultados` bajo la ruta configurada de Drive si está disponible, o junto al Parquet seleccionado. DuckDB puede usar `btc_eda_work` en disco local para archivos temporales.

| Ruta relativa a `OUT` | Contenido |
| --- | --- |
| `entorno.json` | Versiones de Python y de las dependencias principales usadas en la corrida. |
| `seleccion_final_por_variable.csv` | Objetivo, candidatos de ventana y estado de selección por serie. |
| `fourier_spot_10s/n_<n>/spot.parquet` | Trayectorias de varianza y volatilidad por punto interior. |
| `fourier_spot_10s/n_<n>/ventanas.parquet` | Resumen y estado de cada bloque, parámetros elegidos y datos para los contrastes. |
| `fourier_spot_10s/n_<n>/contrastes.csv` y `histograma.png` | Diagnósticos y distribución para esa duración. |
| `fourier_spot_10s/n_60/mz_todos.csv` | Contraste ampliado de calibración para 10 minutos. |
| `fourier_spot_10s/n_<n>/lstm_2_3min.parquet` | Episodios que comparten los dos horizontes. |
| `fourier_spot_10s/n_<n>/lstm_h<h>/` | Modelo Keras, `entrenamiento.csv` y `metricas_test.csv` después del entrenamiento. |

Si falla el **acceso a Drive**, comprobá qué cuenta está autenticada y si puede descargar el Parquet original. Si ya tenés una copia, definí `BTC_PARQUET_PATH` y volvé a ejecutar la primera celda. Si falla el **control temporal**, verificá tipo, unidad, zona y duplicados antes de repetir la preparación. Si aparece **`data10` ausente** tras reiniciar el kernel, repetí las dos primeras celdas de código. Si falta **`ventanas.parquet`** al preparar episodios, ejecutá Fourier para el mismo `n` y comprobá la ruta `OUT` de la sesión.
