import numpy as np
from Model import create_model
from sklearn.metrics import classification_report, accuracy_score
import tensorflow as tf
import pandas as pd

# --- Cargar pesos de usuarios (usa los 5 que tengas) ---
usuarios = ["pesos_user1.npz", "pesos_user2.npz", "pesos_user3.npz", "pesos_user4.npz", "pesos_user5.npz"]

pesos_lista = []
for archivo in usuarios:
    data = np.load(archivo, allow_pickle=True)
    pesos = [data[key] for key in data.files]  # Asegura el orden correcto
    pesos_lista.append(pesos)

# --- Métodos de agregación ---
def fed_avg(pesos_list):
    return [np.mean(p, axis=0) for p in zip(*pesos_list)]

def fed_median(pesos_list):
    return [np.median(p, axis=0) for p in zip(*pesos_list)]

def fed_prox(pesos_list, mu=0.01):
    avg = fed_avg(pesos_list)
    return [w - mu * (w - a) for w, a in zip(avg, avg)]

# --- Evaluar un modelo global ---
def evaluar_modelo(pesos, nombre_metodo):
    modelo = create_model()
    modelo.set_weights(pesos)

    (_, _), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    x_test = x_test / 255.0  

    y_pred = np.argmax(modelo.predict(x_test), axis=1)
    acc = accuracy_score(y_test, y_pred)

    report = classification_report(y_test, y_pred, output_dict=True)
    print(f"Reporte para {nombre_metodo}:")
    print(classification_report(y_test, y_pred))

    return {
        "Método": nombre_metodo,
        "Accuracy": round(acc, 4),
        "F1-score (macro)": round(report["macro avg"]["f1-score"], 4)
    }

# --- Ejecutar todos los métodos ---
resultados = [
    evaluar_modelo(fed_avg(pesos_lista), "FedAvg"),
    evaluar_modelo(fed_median(pesos_lista), "FedMedian"),
    evaluar_modelo(fed_prox(pesos_lista), "FedProx")
]

# --- Mostrar tabla resumen ---
df_resultados = pd.DataFrame(resultados)
print("Resumen comparativo:")
print(df_resultados)

