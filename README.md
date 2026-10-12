# Act 4: Código de experimentación base con MLFlow
  
El presente `README.md` describe como instalar las dependencias necesarias para correr el script de entrenamiento del modelo.
  
  
- [Instalar dependencias](#instalar-dependencias )
- [Ejecutar el script](#ejecutar-el-script )
  - [Argumentos](#argumentos )
  
# Instalar dependencias
  
Las dependencias de este proyecto se pueden instalar utilizando `pip` y el archivo de `requirements.txt`
  
Un ejemplo de como hacer esto en la terminal de Windows:
``` bash
pip install -r 'requirements.txt'
```
  
# Ejecutar el script
  
El script se puede ejecutar de la siguiente manera:
``` bash
python train.py -r 1 -c 1.0 -s lbfgs -m 100
```
  
O utilizando los nombres mas largos de cada argumento:
``` bash
python train.py --RandomSeed 1 --InvReg 1.0 --Solver lbfgs -MaxIter 100
```
  
## Argumentos
  
Descripción de cada uno de los argumentos disponibles:
| Flag | Verbose | Descripción |
| --- | --- | --- |
| r | RandomSeed | Random seed a utilizar con las funciones que la aceptan como parámetro, para asegurar la reproducibilidad de los experimentos. |
| c | InvReg | Para la regresión logística, el valor del inverso de la fuerza de regularización. |
| s | Solver | Algoritmo a utilizar en la regresión logística. Debe usarse una de las opciones provistas en la documentación de scikit-learn para LogisticRegression. |
| m | MaxIter | Máximo número de iteraciones permitidas para la convergencia del _Solver_. |
  