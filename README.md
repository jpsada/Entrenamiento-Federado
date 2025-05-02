# Proyecto de Aprendizaje Federado con MNIST
## Equipo 5


## Archivos del proyecto

- `Model.py`: contiene la estructura del modelo que usamos en común todos los usuarios.
- `Entrenamiento_Local.ipynb`: cada quien entrena su parte del modelo aquí, guarda los pesos y genera sus gráficas y métricas.
- `GlobalModel.py`: une los modelos individuales en uno solo usando tres métodos diferentes de agregación.

## División de datos (privado)

La base MNIST se dividió en 5 partes iguales (una por persona del equipo). Esa división se hizo fuera del repositorio porque son datos privados. Cada usuario tiene su subconjunto guardado como `.npz` dentro de una carpeta llamada `datos_privados/` que no se sube a Git.

## Métodos de agregación

Se usaron tres formas para construir el modelo global:

- **FedAvg**: promedia los pesos de todos los modelos.
- **FedMedian**: toma la mediana de los pesos en lugar del promedio, para evitar que un usuario con datos muy distintos afecte el modelo.
- **FedProx**: es parecido a FedAvg pero penaliza que los modelos locales se alejen mucho del global. Ayuda si los datos de los usuarios son muy diferentes.

## Cómo se usa

1. Cada quien entrena su parte con su subconjunto ejecutando `Entrenamiento_Local.ipynb`.
2. Se guarda el archivo de pesos (`.npz`) localmente.
3. Se juntan los archivos de todos los usuarios en la misma carpeta.
4. Se ejecuta `GlobalModel.py` para obtener el modelo global y se guarda como `modelo_global.weights.h5`.

## Ensayo: Comparación de métodos de agregación en Aprendizaje Federado

<img width="579" alt="Screenshot 2025-05-02 at 5 54 57 p m" src="https://github.com/user-attachments/assets/24621108-4173-4880-9e80-5e54ce14d5ee" />

En nuestro experimento, **FedAvg** y **FedProx** obtuvieron el mejor desempeño con una accuracy de 0.3731 y un F1-score macro de 0.2624, mientras que **FedMedian** tuvo resultados inferiores. Esto se debe a que FedAvg y FedProx, al promediar los pesos de los modelos locales, lograron mantener mejor la información aprendida, mientras que FedMedian, al usar la mediana, perdió detalles útiles de la distribución.

Dado que el dataset MNIST está relativamente balanceado y las divisiones locales fueron similares, métodos simples como FedAvg fueron suficientes. FedProx no mejoró sobre FedAvg debido a un valor bajo de su parámetro de ajuste, por lo que en este caso, **FedAvg fue el método más eficaz**.

