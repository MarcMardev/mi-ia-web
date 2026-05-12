"""
PRÁCTICA SESIÓN 11 ~ Entrenamiento y Validación de Redes Neuronales
====================================================================
En esta práctica exploramos el ciclo de vida completo de un modelo de
aprendizaje profundo aplicado al reconocimiento de dígitos (MNIST).
Realizamos un análisis forense de errores, visualizamos el proceso de
decisión mediante mapas de calor SHAP y establecemos umbrales de
seguridad para aplicaciones reales.
"""

import os
import tensorflow as tf
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
import shap

# Carpeta donde se guardarán todas las gráficas generadas
os.makedirs("graficas", exist_ok=True)


# =============================================================================
# 1. CONFIGURACIÓN
# =============================================================================

"""
DATOS_ENTRENAMIENTO controla cuántas imágenes del dataset MNIST se usan
para entrenar el modelo. A más datos, mejor generalización pero mayor
tiempo de cómputo. Se han probado los valores: 500, 1000, 5000, 20000
y 50000 para comparar el rendimiento.

NUM_EPOCHS es el número de vueltas completas al conjunto de datos de
entrenamiento. Demasiadas épocas con pocos datos provoca sobreajuste
(overfitting): el modelo memoriza en lugar de aprender.
"""

DATOS_ENTRENAMIENTO = 20000
NUM_EPOCHS = 12


# =============================================================================
# 2. CARGA Y PREPARACIÓN DEL MODELO
# =============================================================================

"""
Cargamos el dataset MNIST, que contiene 60.000 imágenes de dígitos
escritos a mano (0-9) para entrenamiento y 10.000 para test.

Cada imagen es de 28x28 píxeles en escala de grises (valores 0-255).
La normalización divide entre 255.0 para que todos los valores queden
en el rango [0, 1], lo que mejora la estabilidad del entrenamiento.

np.expand_dims añade una dimensión extra al final (axis=-1) para que
la forma sea (28, 28, 1), requerida por la capa Flatten con input_shape.
"""

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

x_train_exp = np.expand_dims(x_train / 255.0, axis=-1)
x_test_exp  = np.expand_dims(x_test  / 255.0, axis=-1)


"""
Definición de la arquitectura de la red neuronal secuencial:

  - Capa 1 (Flatten): Convierte la imagen 2D de 28x28 píxeles en un
    vector plano de 784 valores. No tiene parámetros entrenables.

  - Capa 2 (Dense 32, ReLU): Capa oculta con 32 neuronas y función de
    activación ReLU (Rectified Linear Unit). Introduce no-linealidad
    para que el modelo pueda aprender patrones complejos.

  - Capa 3 (Dense 10, Softmax): Capa de salida con 10 neuronas, una
    por dígito (0-9). Softmax convierte las salidas en probabilidades
    que suman 1, indicando la confianza del modelo en cada clase.

La compilación define:
  - optimizer='adam': algoritmo de optimización adaptativo.
  - loss='sparse_categorical_crossentropy': función de pérdida para
    clasificación multiclase con etiquetas enteras.
  - metrics=['accuracy']: métrica de seguimiento durante el entrenamiento.
"""

model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28, 1)),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)


# =============================================================================
# 3. ENTRENAMIENTO DEL MODELO
# =============================================================================

"""
El entrenamiento ajusta los pesos internos de la red para minimizar la
función de pérdida.

  - Se usan los primeros DATOS_ENTRENAMIENTO ejemplos del conjunto de
    entrenamiento.
  - validation_split=0.2 reserva el 20% de esos datos como conjunto de
    validación (no se usa para entrenar, solo para evaluar).
  - verbose=1 muestra el progreso por pantalla en cada época.

OBSERVACIÓN (DATOS=1000, EPOCHS=10):
  La pérdida (loss) baja y la precisión (accuracy) sube en cada época.
  En entrenamiento el cambio es muy pronunciado porque el modelo se
  optimiza directamente sobre esos datos. En validación el ritmo es más
  lento: el modelo no ha visto esos ejemplos, por lo que refleja la
  capacidad real de generalización.
"""

history = model.fit(
    x_train_exp[:DATOS_ENTRENAMIENTO],
    y_train[:DATOS_ENTRENAMIENTO],
    epochs=NUM_EPOCHS,
    validation_split=0.2,
    verbose=1
)


# =============================================================================
# 4. EVALUACIÓN Y MATRIZ DE CONFUSIÓN
# =============================================================================

"""
Evaluamos el modelo sobre los 10.000 ejemplos de test (nunca vistos).

La Matriz de Confusión es una tabla 10x10 donde:
  - Filas = etiqueta real del dígito.
  - Columnas = predicción del modelo.
  - Diagonal principal = predicciones correctas.
  - Valores fuera de la diagonal = confusiones entre dígitos.

OBSERVACIÓN (DATOS=1000, EPOCHS=10):
  La pareja que más se confunde es 3→5 (102 casos), seguida de 9↔4 y
  8↔5. Visualmente, el 3 y el 5 comparten trazos curvos similares, por
  lo que una caligrafía descuidada puede hacer que el modelo falle.
  Los dígitos 5, 8 y 9 también presentan las métricas de precisión y
  recall más bajas del reporte.
"""

preds = np.argmax(model.predict(x_test_exp), axis=1)

plt.figure(figsize=(8, 6))
sns.heatmap(
    confusion_matrix(y_test, preds),
    annot=True,
    fmt='d',
    cmap='Blues'
)
plt.title("Mapa de aciertos y errores (Matriz de Confusión)")
plt.xlabel("Predicción")
plt.ylabel("Etiqueta real")
plt.tight_layout()
plt.savefig("graficas/04_matriz_confusion.png", dpi=150)
plt.show()

print("\nREPORTE FINAL:")
print(classification_report(y_test, preds))


# =============================================================================
# 5. CÓMO ESTÁ APRENDIENDO EL MODELO
# =============================================================================

"""
La gráfica de curvas de aprendizaje muestra la evolución de la precisión
(accuracy) en entrenamiento y validación a lo largo de las épocas.

PREGUNTA: Si la línea de validación (naranja) se estanca mientras la de
entrenamiento (azul) sigue subiendo → el modelo está memorizando los
datos de entrenamiento (sobreajuste / overfitting). Generalizará mal
ante datos nuevos.

PREGUNTA: Con NUM_EPOCHS=20 la precisión de entrenamiento alcanza el
100%, pero la de validación cae ligeramente. No hemos mejorado el
sistema; hemos empeorado el overfitting.

PREGUNTA (DATOS=20000, EPOCHS=12):
  Con más datos la gráfica muestra ambas líneas creciendo de forma
  paralela y más cercanas entre sí. La brecha entre entrenamiento y
  validación es mucho menor que con 1000 datos, lo que indica una
  mejor generalización. El F1-score supera 0.95.
"""

plt.figure(figsize=(10, 4))
plt.plot(history.history['accuracy'],     label='Entrenamiento', marker='o')
plt.plot(history.history['val_accuracy'], label='Validación',     marker='s')
plt.title("¿Cómo está aprendiendo mi IA?")
plt.gca().xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("graficas/05_curvas_aprendizaje.png", dpi=150)
plt.show()


# =============================================================================
# 6. ANÁLISIS FORENSE SHAP
# =============================================================================

"""
SHAP (SHapley Additive exPlanations) permite entender en qué píxeles
se fija el modelo para tomar su decisión:

  - Píxeles ROJOS (valor SHAP positivo): apoyan la predicción.
  - Píxeles AZULES (valor SHAP negativo): contradicen la predicción.

Se usa DeepExplainer, que es eficiente para redes neuronales de Keras.
Se toman 100 imágenes de entrenamiento como muestra de referencia.

COMPARATIVA (500 datos vs 50000 datos, ambos con 6 épocas):
  - Con 500 datos: el mapa de calor es ruidoso. Los colores están
    dispersos por todo el fondo, el modelo ve "fantasmas" en zonas
    sin trazo. No ha aprendido bien la forma del número.
  - Con 50000 datos: el mapa de calor es mucho más limpio y definido.
    Los píxeles relevantes se concentran sobre el trazo real del número.
    El modelo es más sensible y preciso en su atención.
"""

idx = 0
explainer   = shap.DeepExplainer(model, x_train_exp[:100])
shap_values = explainer.shap_values(x_test_exp[idx:idx+1])

print(f"Número real: {y_test[idx]}")
shap.image_plot(shap_values, x_test_exp[idx:idx+1], show=False)
plt.savefig("graficas/06_shap_ejemplo.png", dpi=150, bbox_inches='tight')
plt.show()


# =============================================================================
# 7. BUSCANDO ERRORES EN EL MODELO
# =============================================================================

"""
Buscamos los casos donde la etiqueta real era 0 pero el modelo predijo
otro dígito. SHAP nos muestra en qué se fijó el modelo en esos errores.

OBSERVACIÓN (DATOS=20000, EPOCHS=12):
  La gran mayoría de errores corresponden a "ceros" con caligrafía
  ambigua: trazos incompletos, forma alargada o inclinada que se asemeja
  a un 6, un 5 o un 2. Un humano también dudaría en muchos de estos
  casos.
  Existe también un subconjunto de errores "tontos": imágenes que
  cualquier persona identificaría correctamente como un 0 pero que el
  modelo clasifica mal, lo que revela limitaciones del modelo actual.
"""

idx_errores = np.where((y_test == 0) & (preds != 0))[0]

if len(idx_errores) > 0:
    sel           = idx_errores[:10]
    shap_vals_err = explainer.shap_values(x_test_exp[sel])
    shap.image_plot(shap_vals_err, x_test_exp[sel], show=False)
    plt.savefig("graficas/07_shap_errores.png", dpi=150, bbox_inches='tight')
    plt.show()
    for i, s in enumerate(sel):
        print(f"Ejemplo {i+1}: Real 0 | IA dijo {preds[s]}")


# =============================================================================
# 8. CONFIANZA Y UMBRAL DE SEGURIDAD
# =============================================================================

"""
En aplicaciones críticas no basta con la predicción: necesitamos saber
cuán seguro está el modelo. Establecemos umbrales de confianza:

  - Si la probabilidad máxima de la predicción >= umbral → la IA decide.
  - Si no llega al umbral → la imagen pasa a revisión humana (Descarte).

Resultados observados (DATOS=20000, EPOCHS=12):
  - Umbral 99%: ~30% de imágenes requieren revisión humana, pero los
    errores automáticos son casi nulos.
  - Umbral 95%: ~15-20% de revisión humana. Buen equilibrio entre
    seguridad y carga operativa. OPCIÓN RECOMENDADA para entornos
    críticos con personal limitado.
  - Umbral 90%: ~10-15% de revisión. Ligero aumento de error, pero
    reduce aún más la carga humana. Válido si el personal es muy escaso.

Conclusión: a mayor umbral, menos errores automáticos pero más carga
humana. El umbral óptimo depende del coste relativo entre un error de
la IA y el coste de la revisión humana.
"""

probs         = model.predict(x_test_exp, verbose=0)
conf          = np.max(probs, axis=1)
preds_conf    = np.argmax(probs, axis=1)
correctos     = (preds_conf == y_test)

umbrales = [0.99, 0.95, 0.90, 0.80, 0.70, 0.50, 0.20]
res = []
for u in umbrales:
    pasan    = conf >= u
    n        = np.sum(pasan)
    aciertos = np.sum(correctos[pasan]) if n > 0 else 0
    res.append([
        u * 100,
        aciertos / 100,
        (n - aciertos) / 100,
        (10000 - n) / 100,
        (aciertos / n * 100 if n > 0 else 100)
    ])

df = pd.DataFrame(res, columns=['U', 'A', 'F', 'D', 'P'])

fig, ax1 = plt.subplots(figsize=(10, 5))
x = range(len(df))
ax1.bar(x, df['A'], color='#2ecc71', label='Aciertos')
ax1.bar(x, df['F'], bottom=df['A'], color='#e74c3c', label='Errores')
ax1.bar(x, df['D'], bottom=df['A'] + df['F'], color='#dfe6e9', label='Duda (Humano)')

ax2 = ax1.twinx()
ax2.plot(x, df['P'], color='#2980b9', marker='o', label='Calidad IA')

ax1.set_xticks(x)
ax1.set_xticklabels([f"{int(i)}%" for i in df['U']])
ax1.set_title("SEGURIDAD: PRECISIÓN VS AUTOMATIZACIÓN")
ax1.legend(loc='upper left', bbox_to_anchor=(1.15, 1))
ax2.legend(loc='upper left', bbox_to_anchor=(1.15, 0.85))
plt.tight_layout()
plt.savefig("graficas/08_confianza_umbrales.png", dpi=150)
plt.show()

# Guardamos el modelo entrenado en formato (.keras)
model.save('modelo_mnist.keras')