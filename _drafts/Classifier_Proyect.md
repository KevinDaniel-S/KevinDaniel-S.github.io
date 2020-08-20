---
layout: post
title:  "Proyecto Clasificación"
date:   2020-08-19 19:35:00 -0500
categories: Machine Learning
---
Modelo de machine learning que distingue números escritos a mano de un conjunto de
datos llamado mnist con 70,000 imágenes, cada una representa un número del 0 al 9, 
están en una resolución 28x28

<!--more-->

Podemos descargar el conjunto de datos mnist desde sklearn.

```python
from sklearn.datasets import fetch_openml
mnist = fetch_openml('mnist_784', version=1)
mnist.keys()
```

|dict_keys|data|target|frame|feature_names|target_names|DESCR|details|categories|url|

Separaremos los vectores que contienen la información de cada imagen de la etiqueta que
los identifica.

```python
X, y = mnist["data"], mnist["target"]
print(X.shape)
print(y.shape)
```

|X.shape|(70000,784)|
|y.shape|(70000,)|

Ya que cada imagen está en forma de vector, podemos redimensionarlo de tal forma 
que tenga las proporciones adecuadas, en este caso es una matriz 28x28, una vez
tenga la forma correcta podemos crear la imagen, miremos la primera.

```python
import matplotlib as mpl
import matplotlib.pyplot as plt

some_digit = X[0]
some_digit_image = some_digit.reshape(28, 28)

plt.imshow(some_digit_image, cmap = mpl.cm.binary, interpolation="nearest")
plt.axis("off")
plt.show()
```

![png](\images\classification\output_3_0.png)

En este caso parece ser un 5, si le preguntamos al conjunto de datos por su valor,
podemos verificar su valor.

|y[0]|'5'|

Al parecer realmente era un 5, pero está en un formato de cadena texto, para
facilitarnos las cosas pasaremos las etiquetas "y" a números enteros.

```python
y = y.astype(np.uint8)
```

Podemos crear una función que directamente nos cree la imagen del número deseado.

```python
def plot_digit(data):
    image = data.reshape(28, 28)
    plt.imshow(image, cmap = mpl.cm.binary,
               interpolation="nearest")
    plt.axis("off")
```

También crearemos una función que nos permita crear varias imágenes juntas.

```python
def plot_digits(instances, images_per_row=10, **options):
    size = 28
    images_per_row = min(len(instances), images_per_row)
    images = [instance.reshape(size,size) for instance in instances]
    n_rows = (len(instances) - 1) // images_per_row + 1
    row_images = []
    n_empty = n_rows * images_per_row - len(instances)
    images.append(np.zeros((size, size * n_empty)))
    for row in range(n_rows):
        rimages = images[row * images_per_row : (row + 1) * images_per_row]
        row_images.append(np.concatenate(rimages, axis=1))
    image = np.concatenate(row_images, axis=0)
    plt.imshow(image, cmap = mpl.cm.binary, **options)
    plt.axis("off")
```

Mostraremos los primeros 100 números que están en el conjunto de datos

```python
plt.figure(figsize=(9,9))
example_images = X[:100]
plot_digits(example_images, images_per_row=10)
plt.show()
```

![png](\images\classification\output_8_0.png)

Ahora separaremos el conjunto de entrenamiento del conjunto de prueba, ya que el
conjunto de datos ya se encuentra revuelto no hay necesidad de dividirlo aleatoriamente
y simplemente podemos partirlo en dos, la primera parte tendrá 60,000 imágenes y la
segunda tendrá las 10,000 restantes.

```python
X_train, X_test, y_train, y_test = X[:60000], X[60000:], y[:60000], y[60000:]
```

# Binary classifier 

Comenzaremos por crear un clasificador binario que distinga a los números que son 5 
de los que no.

```python
y_train_5 = (y_train == 5)
y_test_5 = (y_test == 5)
```

Para empezar entrenaremos el algoritmo del descenso de gradiente estocástico.

```python
from sklearn.linear_model import SGDClassifier

sgd_clf = SGDClassifier(max_iter=1000, tol=1e-3, random_state=42)
sgd_clf.fit(X_train, y_train_5)
```

Ahora que lo tenemos entrenado podemos probarlo con el número que ya conocemos.

|sgd_clf.predict([some_digit])|array([ True])|

Al parecer lo identificó correctamente como un 5

# Medir el rendimiento

Intentaremos verificar su rendimiento, para eso usarémos cross validation, que
dividirá el conjunto de entrenamiento en 3 partes, entrenará al modelo con dos
tercias partes del conjunto y lo validará con la parte restante, esto lo hará
3 veces, cada vez dejando una parte diferente para la validación.


```python
from sklearn.model_selection import cross_val_score
cross_val_score(sgd_clf, X_train, y_train_5, cv=3, scoring="accuracy")
```

|0.95035|0.96035|0.9604|

Al parecer acertó más del 95% de veces en las tres ocasiones que se probó el
modelo, sin embargo no podemos emocionarnos tan fácilmente, antes tenemos que
comprobar un clasificador tonto, que clasificará a todos los números como no 5.

```python
from sklearn.base import BaseEstimator

class Never5Classifier(BaseEstimator):
    def fit(self, X, y=None):
        pass
    def predict(self, X):
        return np.zeros((len(X), 1), dtype=bool)
```

Intentemos probar su rendimiento.

```python
never_5_clf = Never5Classifier()
cross_val_score(never_5_clf, X_train, y_train_5, cv=3, scoring="accuracy")
```

|0.91125|0.90855|0.90915|


Al parecer acertó más del 90% de ocasiones, solo fallando cuando el número realmente vale
5, esto demuestra que no podemos fiarnos ciegamente de la exactitud.

## Matriz de confusión

Algo que nos permitirá evaluar mejor el rendimiento son las matrices de confusión,
esta vez usaremos un algoritmo similar al anterior, pero en lugar de darnos un
puntaje, nos dará las predicciones.

```python
from sklearn.model_selection import cross_val_predict

y_train_pred = cross_val_predict(sgd_clf, X_train, y_train_5, cv=3)
```

Ahora que tenemos las predicciones podemos crear la matriz de confusión

```python
from sklearn.metrics import confusion_matrix

confusion_matrix(y_train_5, y_train_pred)
```

|Real\Predicción|No 5|5|
|No 5|53892|687|
|5|1891|3530|

La primera fila representa a los números que no son 5, mientras que la segunda fila
representa a los números que sí son 5; La primera columna representa los  números que 
el modelo interpretó como no 5 y la segunda columna representa a los números que el 
modelo interpretó como 5.

53,892 números fueron clasificados correctamente como no 5, llamados verdaderos negativos.

3,530 números fueron clasificados correctamente como 5, llamados verdaderos positivos.

687 números fueron clasificados incorrectamente como 5, llamados falsos positivos.

1891 números fueron clasificados incorrectamente como no 5, llamados falsos negativos.

Un clasificador perfecto solo tendría verdaderos positivos o verdaderos negativos, por
lo que solo tendría valores diferentes de 0 en su diagonal principal.

## Precisión y exhaustividad

La matriz de confusión por si sola no nos dice mucho, para eso tenemos que usar dos
métricas más consisas 

La precisión se calcula los positivos verdaderos sobre el total de predicciones
positivas, en este caso los positivos verdaderos serían 3530, mientras que el total
de predicciones positivas serían 3530 + 687


```python
from sklearn.metrics import precision_score, recall_score
```

|precision_score(y_train_5, y_train_pred)|0.837|

La exhaustividad se calcula como el número de positivos verdaderos sobre el total de
positivos, en este caso los positivos verdaderos serían 3530, mientras que el total de
positivos serían 3530 + 1891

|recall_score(y_train_5, y_train_pred)|0.651|

Es común combinar ambas métricas en una sola llamada F<sub>1</sub>, consiste en la media
armonica entre las dos métricas.

![F1](\images\classification\armonica.png)

```python
from sklearn.metrics import f1_score
```

|f1_score(y_train_5, y_train_pred)|0.732|

El puntaje F<sub>1</sub> beneficia a aquellos modelos que tienen precisión y
exhaustividad similares, sin embargo en algunos contextos te interesa más la
precisión que la exhaustividad o viceversa, desafortunadamente no puedes tener 
ambas a la vez, ya que si aumentas la precisión, la exhaustividad se verá afectada,
a esto se le llama compensación precisión/exhaustividad

## Compensación precisión/exhaustividad

Para comprender esta compensación miremos como el clasificador sgd toma las decisiones.
Por cada instancia calcula un puntaje basado en la función de decisión, si el
puntaje que obtuvimos es mayor que el umbral, entonces lo clasifica como positivo, en
caso contrario lo clasifica como negativo, dependiendo del umbral que coloquemos nuestra
precisión y exhaustividad se verán afectadas.

![](\images\classification\tradeoff.png)

Probemos esto con el dígito que ya conocemos, primero veremos su puntaje

```python
y_scores = sgd_clf.decision_function([some_digit])
```

|y_scores|2164.22030239|

Ahora modificamos el umbral para que sea igual a 0 y miraremos como lo clasifica

```python
threshold = 0
y_some_digit_pred = (y_scores > threshold)
```

|y_some_digit_pred|True|

Lo clasificó como positivo

Ahora modificaremos el umbral para que sea 8000 y ver como lo clasifica

```python
threshold = 8000
y_some_digit_pred = (y_scores > threshold)
```

|y_some_digit_pred|False|

Ya que el puntaje es menor que el umbral entonces lo clasificó como negativo

Para decidir el umbral que usaremos, podemos usar cross validation 

```python
y_scores = cross_val_predict(sgd_clf, X_train, y_train_5, cv=3,
                             method="decision_function")
```

Ahora podemos calcular la presición y la exhaustividad para todos los valores 
positivos del umbral.

```python
from sklearn.metrics import precision_recall_curve

precisions, recalls, thresholds = precision_recall_curve(y_train_5, y_scores)
```

Ahora se hará una gráfica de la precisión y la exhaustividad por todos los valores 
positivos del umbral

```python
def plot_precision_recall_vs_threshold(precisions, recalls, thresholds):
    plt.plot(thresholds, precisions[:-1], "b--", label="Precision", linewidth=2)
    plt.plot(thresholds, recalls[:-1], "g-", label="Recall", linewidth=2)
    plt.legend(loc="center right", fontsize=16) 
    plt.xlabel("Threshold", fontsize=16)        
    plt.grid(True)                              
    plt.axis([-50000, 50000, 0, 1])             

recall_90_precision = recalls[np.argmax(precisions >= 0.90)]
threshold_90_precision = thresholds[np.argmax(precisions >= 0.90)]


plt.figure(figsize=(8, 4))                                                                  
plot_precision_recall_vs_threshold(precisions, recalls, thresholds)
plt.plot([threshold_90_precision, threshold_90_precision], [0., 0.9], "r:")                 
plt.plot([-50000, threshold_90_precision], [0.9, 0.9], "r:")                                
plt.plot([-50000, threshold_90_precision], [recall_90_precision, recall_90_precision], "r:")
plt.plot([threshold_90_precision], [0.9], "ro")                                             
plt.plot([threshold_90_precision], [recall_90_precision], "ro")                             
plt.show()                                            
```

![png](\images\classification\output_32_0.png)

Podemos apreciar que la curva de precisión se ve más accidentada que la curva de
exhaustividad, esto se debe a que algunas veces la precisión caerá cuando el
umbral es aumentado, sin embargo la exhaustividad solo puede bajar cuando se 
aumenta el umbral.

Para elegir el umbral perfecto también podemos crear una gráfica
la exhaustividad contra la precisión.

```python
def plot_precision_vs_recall(precisions, recalls):
    plt.plot(recalls, precisions, "b-", linewidth=2)
    plt.xlabel("Recall", fontsize=16)
    plt.ylabel("Precision", fontsize=16)
    plt.axis([0, 1, 0, 1])
    plt.grid(True)

plt.figure(figsize=(8, 6))
plot_precision_vs_recall(precisions, recalls)
plt.plot([0.4368, 0.4368], [0., 0.9], "r:")
plt.plot([0.0, 0.4368], [0.9, 0.9], "r:")
plt.plot([0.4368], [0.9], "ro")
plt.show()
```

![png](\images\classification\output_33_0.png)

Podemos apreciar que la precisión comienza a caer drásticamente cuando la exhaustividad
es cercana a 0.80, lo más probable es que quieras utilizar un umbral antes de que eso
suceda

Entonces supongamos que quieres una precisión del 90%, entonces miras la gráfica y
te das cuenta de que necesitas un umbral de aproximadamente 8,000, para ser más 
precisos, buscaremos el umbral más bajo que nos de al menos 90% de precisión

```python
threshold_90_precision = thresholds[np.argmax(precisions >= 0.90)]
```

|threshold_90_precision|7816|

Ahora haremos las predicciones

```python
y_train_pred_90 = (y_scores >= threshold_90_precision)
```

Miraremos cuales son los puntajes de la precisión y de la exhaustividad con ese
umbral

|precision_score(y_train_5, y_train_pred_90)|0.90003|
|recall_score(y_train_5, y_train_pred_90)|0.47998|

Podemos observar que nuestra precisión llegó a 90% satisfactoriamente.

## Curva Roc

Otra herramienta que podemos utilizar para medir el rendimiento de un clasificador
binario es la curva *receiver operating characteristic* (ROC). Es muy similar 
a la curva exhaustividad/precisión, pero en lugar de gráficar precisión contra 
exhaustividad, gráficaremos el ratio de positivos verdaderos (Otro nombre para la
exhaustividad) contra el ratio de falsos positivos, esto ultimo es el ratio de
instancias negativas que fueron clasificadas incorrectamente como positivas.

```python
from sklearn.metrics import roc_curve

fpr, tpr, thresholds = roc_curve(y_train_5, y_scores)
```

```python
def plot_roc_curve(fpr, tpr, label=None):
    plt.plot(fpr, tpr, linewidth=2, label=label)
    plt.plot([0, 1], [0, 1], 'k--') 
    plt.axis([0, 1, 0, 1])                                    
    plt.xlabel('False Positive Rate (Fall-Out)', fontsize=16) 
    plt.ylabel('True Positive Rate (Recall)', fontsize=16)    
    plt.grid(True)                                            

plt.figure(figsize=(8, 6))                         
plot_roc_curve(fpr, tpr)
plt.plot([4.837e-3, 4.837e-3], [0., 0.4368], "r:") 
plt.plot([0.0, 4.837e-3], [0.4368, 0.4368], "r:")  
plt.plot([4.837e-3], [0.4368], "ro")               

plt.show()
```

![png](\images\classification\output_40_0.png)

Volvemos a ver una compensación entre mayor sea la exhaustividad, producirá más
falsos positivos. La línea punteada representa la curva ROC de un clasificador aleatorio
para medir el rendimiento de un modelo usando la curva ROC es asegurarse de que la curva
se aleje lo más posible de esa línea punteada (Hacia la esquina superior izquierda) 

Una forma de comparar los diferentes modelos de clasificación es medir el area bajo
la curva (AUC), un clasificador perfecto tendrá una area bajo la curva igual a 1,
mientras que en un clasificador aleatorio será aproximadamente 0.5.

```python
from sklearn.metrics import roc_auc_score
```

|roc_auc_score(y_train_5, y_scores)|0.9604938554008616|

Vamos a entrenar un clasificador de bosque aleatorio y compararemos con la curva ROC
y el ROC AUC del clasificador SGD.

Ya que bosque aletorio no tiene un método de decisión, usaremos el método 
predict_proba(), que nos devuelve un vector por cada instancia conteniendo la
probabilidad de pertenencia a cada clase.

```python
from sklearn.ensemble import RandomForestClassifier

forest_clf = RandomForestClassifier(random_state=42)
y_probas_forest = cross_val_predict(forest_clf, X_train, y_train_5, cv=3,
                                    method="predict_proba")
```

Ya que para la curva ROC necesitamos puntajes y no probabilidades, usaremos la
probabilidad de pertenencia a la clase positiva como puntaje.

```python
y_scores_forest = y_probas_forest[:, 1]

fpr_forest, tpr_forest, thresholds_forest = roc_curve(y_train_5,y_scores_forest)
```

Ahora ya podemos gráficar la curva ROC

```python
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, "b:", linewidth=2, label="SGD")
plot_roc_curve(fpr_forest, tpr_forest, "Random Forest")
plt.plot([4.837e-3, 4.837e-3], [0., 0.4368], "r:")
plt.plot([0.0, 4.837e-3], [0.4368, 0.4368], "r:")
plt.plot([4.837e-3], [0.4368], "ro")
plt.plot([4.837e-3, 4.837e-3], [0., 0.9487], "r:")
plt.plot([4.837e-3], [0.9487], "ro")
plt.grid(True)
plt.legend(loc="lower right", fontsize=16)

plt.show()
```

![png](\images\classification\output_44_0.png)

Como podemos observar, el modelo de bosque aleatorio rindió mejor que el modelo
SGD

## Clasificación multiclase

Mientras que el clasificador binario solo pueden distinguir entre dos clases, el
clasificador multiclase puede distinguir entre más de dos clases.

Algunos algoritmos son capaces de manejar la clasificación multiclase directamente,
mientras que otros son estrictamente binarios. Sin embargo hay algunas estrategías
que se pueden aplicar para que un clasificador binario pueda clasificar multiples clases

Una forma de crear un sistema que clasifique 10 clases diferentes es crear un detector
para cada clase, uno para cada digito, entonces cuando quieras clasificar una imagen,
obtienes el puntaje de pertenencia a cada clase, entonces eliges la clase que obtuvo
un puntaje mayor. Esto es llamado One vs All (OvA)

Otra estrategia es entrenar un modelo para cada par de dígitos, uno que distinga al
0 del 1, otro que distinga el 0 del 2 y así sucesivamente. Esto es llamado One vs One
(OvO). Si hay N clases, necesitas entrenar N * (N - 1) / 2 clasificadores. Para este
problema se necesitarían 45 clasificadores, entonces utilizaríamos la clase que
ganase más duelos


Para esto entrenaremos el clasificador SVC, por defecto usará (OvO) y miraremos su 
predicción

```python
from sklearn.svm import SVC

svm_clf = SVC(gamma="auto", random_state=42)
svm_clf.fit(X_train[:1000], y_train[:1000])
svm_clf.predict([some_digit])
```

|[5]|dtype=uint8|

El modelo clasificó satisfactoriamente el dígito, ahora imprimiremos los puntajes
de cada clase

```python
some_digit_scores = svm_clf.decision_function([some_digit])
some_digit_scores
```

|0|1|2|3|4|5|6|7|8|9|
|2.81|7.09|3.82|0.79|5.88|9.29|1.79|8.10|-0.22|4.83|

A simple vista podemos comprobar que la clase con un mayor puntaje es el 5

Si queremos forzar al modelo a utilizar la estrategia OvA, podemos usar el constructor
OneVsRestClassifier

```python
from sklearn.multiclass import OneVsRestClassifier
ovr_clf = OneVsRestClassifier(SVC(gamma="auto", random_state=42))
ovr_clf.fit(X_train[:1000], y_train[:1000])
ovr_clf.predict([some_digit])
```

|[5]|dtype=uint8|

Lo predijo satisfactoriamente como 5, ahora veamos cuantos estimadores utilizó

|len(ovr_clf.estimators_)|10|

Como lo vimos anteriormente usaría 10 con la estrategía OvA

Entrenar un bosque aleatorio es igual de sencillo.

```python
forest_clf.fit(X_train, y_train)
forest_clf.predict([some_digit])
```

|[5]|dtype=uint8|

Miraremos la probabilidad de pertenencia a cada clase

```python
forest_clf.predict_proba([some_digit])
```

|0|1|2|3|4|5|6|7|8|9|
|0.00|0.00|0.01|0.08|0.00|0.90|0.00|0.00|0.00|0.01|

La clase que tiene una mayor probabilidad es el 5, por lo tanto el algoritmo elige
esa clase.

Para medir el rendimiento de los clasificadores utilizaremos cross validation

```python
cross_val_score(sgd_clf, X_train, y_train, cv=3, scoring="accuracy")
```

|0.87365|0.85835|0.8689|

El algoritmo SGD rindió con un puntaje aproximado de 86% de exactitud

Ahora miraremos el puntaje del bosque aleatorio

```python
cross_val_score(forest_clf, X_train, y_train, cv=3, scoring="accuracy")
```

|0.9646|0.96255|0.9666|

El bosque aleatorio rindió mucho mejor que el SGD

Si queremos mejorar nuestros puntajes podemos estandarizar los valores

```python
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train.astype(np.float64))
```

Ahora que están estandarizados podemos volver a aplicar la validación

```python
cross_val_score(sgd_clf, X_train_scaled, y_train, cv=3, scoring="accuracy")
```

|0.8983|0.891|0.9018|

El algoritmo SGD mejoró un poco

```python
cross_val_score(forest_clf, X_train_scaled, y_train, cv=3, scoring="accuracy")
```

|0.96445|0.96255|0.96645|

Mientras que el bosque no se vió muy afectado

## Analisis de error

Si queremos mejorar nuestro modelo podemos analizar los diferentes errores que
cometé, esto lo podemos lograr utilizando una matriz de confusión con todas las
predicciones

```python
y_train_pred = cross_val_predict(forest_clf, X_train_scaled, y_train, cv=3)
conf_mx = confusion_matrix(y_train, y_train_pred)
conf_mx
```

|5840|   1|   8|   2|   4|   9|  20|   1|  35|   3|
|   1|6634|  43|  12|  13|   5|   6|  13|  12|   3|
|  26|  12|5749|  29|  32|   5|  20|  37|  42|   6|
|   7|   7|  93|5809|   3|  63|   7|  49|  61|  32|
|  12|  13|  14|   1|5643|   0|  29|  14|  17|  99|
|  20|   9|   9|  65|  13|5195|  53|   6|  32|  19|
|  25|  11|   5|   0|  12|  45|5805|   0|  15|   0|
|   4|  24|  58|   6|  37|   1|   0|6037|  11|  87|
|   9|  35|  44|  53|  26|  52|  27|   5|5524|  76|
|  21|  10|  13|  76|  75|  15|   3|  58|  45|5633|

Los números que se encuentran en la diagonal principal son aquellos que clasificó
correctamente.

Para apreciar mejor la matriz de confusión podemos utilizar un mapa de calor.

```python
import seaborn as sns; sns.set()
ax = sns.heatmap(conf_mx, cmap='gist_heat', linewidths=.5)
```

![png](\images\classification\output_63_0.png)

Entre más claro sea una casilla más valores pertenecen a esa casilla, como podemos
observar la diagonal principal es de un color más brillante que el resto de la
matriz.

Si solo nos interesan los valores incorrectos que tuvimos, entonces normalizaremos
la matriz y volveremos 0's la matriz principal

```python
row_sums = conf_mx.sum(axis=1, keepdims=True)
norm_conf_mx = conf_mx / row_sums
np.fill_diagonal(norm_conf_mx, 0)
ax = sns.heatmap(norm_conf_mx, cmap='gist_heat', linewidths=.5)
```

![png](\images\classification\output_65_0.png)

Podemos apreciar que el mayor número de error es cuando es un 4, pero el modelo lo
clasifica como 9; También cuando es un 3, pero el modelo lo clasifica como un 2.

Miraremos de cerca los 4 y los 9

```python
cl_a, cl_b = 4, 9
X_aa = X_train[(y_train == cl_a) & (y_train_pred == cl_a)]
X_ab = X_train[(y_train == cl_a) & (y_train_pred == cl_b)]
X_ba = X_train[(y_train == cl_b) & (y_train_pred == cl_a)]
X_bb = X_train[(y_train == cl_b) & (y_train_pred == cl_b)]

plt.figure(figsize=(8,8))
plt.subplot(221); plot_digits(X_aa[:25], images_per_row=5)
plt.subplot(222); plot_digits(X_ab[:25], images_per_row=5)
plt.subplot(223); plot_digits(X_ba[:25], images_per_row=5)
plt.subplot(224); plot_digits(X_bb[:25], images_per_row=5)
plt.show()
```

Los números que se encuentran en la parte de arriba son aquellos que son 4, mientras 
que los que están abajo son 9. De mismo modo, los números que están a la izquierda 
son aquellos que fueron clasificados como 4, mientras que los números a la derecha 
son aquellos que fueron clasificados como 9.

![png](\images\classification\output_66_0.png)

Entonces el primer bloque de arriba son aquellos números que fueron clasificados como
4 y realmente son 4.

El segundo bloque de arriba son aquellos que fueron clasificados como 9, pero realmente
son 4.

Los números del primer bloque de abajo son aquellos que fueron clasificados como 4, pero
realmente son 9.

Los números del segundo bloque de abajo son aquellos que fueron clasificados como 9 
y realmente son 9.

Podemos apreciar que unos números son difíciles de distinguir inclusive para humanos.

## Clasificación multi-etiqueta

Algunas veces queremos clasificar un objetivo con varias etiquetas a la vez, por
ejemplo cuando queremos identificar los objetos en una fotografía.

Para este ejemplo crearemos un modelo que identifique a los números impares y los
números mayores que 6. Usando el algoritmo KNN.

```python
from sklearn.neighbors import KNeighborsClassifier

y_train_large = (y_train >= 7)
y_train_odd = (y_train % 2 == 1)
y_multilabel = np.c_[y_train_large, y_train_odd]

knn_clf = KNeighborsClassifier()
knn_clf.fit(X_train, y_multilabel)
```

Ahora que tenemos el modelo entrenado, clasificaremos el valor que anteriormente
tomamos como ejemplo

```python
knn_clf.predict([some_digit])
```

|False|True|

Como resultado nos dio un vector con dos valores booleanos, el primero nos dice
si es un valor mayor que 6, ya que 5 es menor que 6 nos dio Falso; El segundo
valor booleano nos dice si 5 es impar, como sí es impar entonces devuelve Verdadero

## Clasificación de salida múltiple

El último tipo de clasificación que veremos, es cuando asignamos a cada muestra un
conjunto de salidas.

Para mostrar esto crearemos un modelo que remueva el ruido en una imagen. Tomará
como entrada la imagen de un dígito con ruido y como salida devolverá una imagen
limpia.

Primero modificaremos las imágenes de dígitos con un ruido generado aleatoriamente.

```python
noise = np.random.randint(0, 100, (len(X_train), 784))
X_train_mod = X_train + noise
noise = np.random.randint(0, 100, (len(X_test), 784))
X_test_mod = X_test + noise
y_train_mod = X_train
y_test_mod = X_test
```

Ahora mostraremos dos números

```python
some_index = 0
plt.subplot(121); plot_digit(X_test_mod[some_index])
plt.subplot(122); plot_digit(y_test_mod[some_index])
plt.show()
```

El número de la izquierda es el número con ruido, mientras que la imagen de la derecha,
es el número sin ruido

![png](\images\classification\output_72_0.png)

Ahora veamos como se comporta nuestro modelo

```python
knn_clf.fit(X_train_mod, y_train_mod)
clean_digit = knn_clf.predict([X_test_mod[some_index]])
plot_digit(clean_digit)
```

![png](\images\classification\output_73_0.png)

Al parecer se acercó mucho a la imagen original, por lo tanto podemos decir que
el modelo tuvo exito.

Para ver el repositorio completo [¡Click aquí!](https://github.com/KevinDaniel-S/MachineLearning/tree/master/Proyecto%20de%20clasificaci%C3%B3n)