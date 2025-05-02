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

