import numpy as np
from Model import create_model  # Asegúrate que el nombre es correcto

# --- Cargar pesos guardados por los usuarios desde .npz ---
pesos_npz = np.load("pesos_usuario.npz", allow_pickle=True)
pesos_usuario1 = [pesos_npz[key] for key in pesos_npz]

# Si hay más usuarios, repite lo anterior para cada uno y colócalo en la lista
pesos_lista = [pesos_usuario1]

# --- FedAvg ---
def fed_avg(pesos_list):
    return [np.mean(p, axis=0) for p in zip(*pesos_list)]

# --- FedMedian ---
def fed_median(pesos_list):
    return [np.median(p, axis=0) for p in zip(*pesos_list)]

# --- FedProx ---
def fed_prox(pesos_list, mu=0.01):
    avg = fed_avg(pesos_list)
    return [w - mu * (w - a) for w, a in zip(avg, avg)]

# Aplicar agregación (puedes cambiar por fed_median o fed_prox)
pesos_globales = fed_avg(pesos_lista)

# Crear modelo y asignar pesos globales
modelo_global = create_model()
modelo_global.set_weights(pesos_globales)

# Guardar
modelo_global.save_weights("modelo_global.weights.h5")
print("Modelo global guardado como 'modelo_global.weights.h5'")

