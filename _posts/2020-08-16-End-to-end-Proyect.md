---
layout: post
title:  "Proyecto de punta a punta del libro Hands on machine learning"
date:   2020-08-17 20:12:00 -0500
categories: Machine Learning
---

Modelo de Machine Learning que ayuda a predecir el valor de una casa en
California dependiendo ciertos factores

<!--more-->

## Descripción del conjunto de datos

Primeras 5 filas del conjunto de datos
```python
housing = pd.read_csv("data/housing.csv")
housing.head()
```
<div class="table-wrapper">

  <style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
  </style>

  <table border="1" class="dataframe">
    <thead>
      <tr style="text-align: right;">
        <th></th>
        <th>longitude</th>
        <th>latitude</th>
        <th>housing_median_age</th>
        <th>total_rooms</th>
        <th>total_bedrooms</th>
        <th>population</th>
        <th>households</th>
        <th>median_income</th>
        <th>median_house_value</th>
        <th>ocean_proximity</th>
      </tr>
    </thead>
    <tbody>
      <tr>
        <th>0</th>
        <td>-122.23</td>
        <td>37.88</td>
        <td>41.0</td>
        <td>880.0</td>
        <td>129.0</td>
        <td>322.0</td>
        <td>126.0</td>
        <td>8.3252</td>
        <td>452600.0</td>
        <td>NEAR BAY</td>
      </tr>
      <tr>
        <th>1</th>
        <td>-122.22</td>
        <td>37.86</td>
        <td>21.0</td>
        <td>7099.0</td>
        <td>1106.0</td>
        <td>2401.0</td>
        <td>1138.0</td>
        <td>8.3014</td>
        <td>358500.0</td>
        <td>NEAR BAY</td>
      </tr>
      <tr>
        <th>2</th>
        <td>-122.24</td>
        <td>37.85</td>
        <td>52.0</td>
        <td>1467.0</td>
        <td>190.0</td>
        <td>496.0</td>
        <td>177.0</td>
        <td>7.2574</td>
        <td>352100.0</td>
        <td>NEAR BAY</td>
      </tr>
      <tr>
        <th>3</th>
        <td>-122.25</td>
        <td>37.85</td>
        <td>52.0</td>
        <td>1274.0</td>
        <td>235.0</td>
        <td>558.0</td>
        <td>219.0</td>
        <td>5.6431</td>
        <td>341300.0</td>
        <td>NEAR BAY</td>
      </tr>
      <tr>
        <th>4</th>
        <td>-122.25</td>
        <td>37.85</td>
        <td>52.0</td>
        <td>1627.0</td>
        <td>280.0</td>
        <td>565.0</td>
        <td>259.0</td>
        <td>3.8462</td>
        <td>342200.0</td>
        <td>NEAR BAY</td>
      </tr>
    </tbody>
  </table>

</div>


Información del conjunto de datos

```python
housing.info()
```

|#|Column            |Non-Null Count|Dtype  |
|-|:----------------:|:------------:|:-----:|
|0|longitude         |20640 non-null|float64|
|1|latitude          |20640 non-null|float64|
|2|housing_median_age|20640 non-null|float64|
|3|total_rooms       |20640 non-null|float64|
|4|total_bedrooms    |20433 non-null|float64|
|5|population        |20640 non-null|float64|
|6|households        |20640 non-null|float64|
|7|median_income     |20640 non-null|float64|
|8|median_house_value|20640 non-null|float64|
|9|ocean_proximity   |20640 non-null|object |
||||       **dtypes**: float64(9), object(1)|

Podemos observar que la columna "total_bedrooms" tiene 207 valores nulos
y la columna "ocean_proximity" es de tipo objeto, ya que leímos los datos
desde un csv podemos inferir que son de tipo texto.

Veremos el contenido de la columna "ocean_proximity"

```python
housing['ocean_proximity'].value_counts()
```

|<1H OCEAN |9136|
|INLAND    |6551|
|NEAR OCEAN|2658|
|NEAR BAY  |2290|
|ISLAND    |5   |
|**Name**: ocean_proximity| **dtype**: int64 |    

Al parecer la columna "ocean_proximity" es una variable categórica de
5 niveles

Ahora generaremos las estadísticas descriptivas del conjunto de datos 

```python
housing.describe()
```

<div class="table-wrapper">
  <style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
  </style>

  <table border="1" class="dataframe">
    <thead>
      <tr style="text-align: right;">
        <th></th>
        <th>longitude</th>
        <th>latitude</th>
        <th>housing_median_age</th>
        <th>total_rooms</th>
        <th>total_bedrooms</th>
        <th>population</th>
        <th>households</th>
        <th>median_income</th>
        <th>median_house_value</th>
      </tr>
    </thead>
    <tbody>
      <tr>
        <th>count</th>
        <td>20640.000000</td>
        <td>20640.000000</td>
        <td>20640.000000</td>
        <td>20640.000000</td>
        <td>20433.000000</td>
        <td>20640.000000</td>
        <td>20640.000000</td>
        <td>20640.000000</td>
        <td>20640.000000</td>
      </tr>
      <tr>
        <th>mean</th>
        <td>-119.569704</td>
        <td>35.631861</td>
        <td>28.639486</td>
        <td>2635.763081</td>
        <td>537.870553</td>
        <td>1425.476744</td>
        <td>499.539680</td>
        <td>3.870671</td>
        <td>206855.816909</td>
      </tr>
      <tr>
        <th>std</th>
        <td>2.003532</td>
        <td>2.135952</td>
        <td>12.585558</td>
        <td>2181.615252</td>
        <td>421.385070</td>
        <td>1132.462122</td>
        <td>382.329753</td>
        <td>1.899822</td>
        <td>115395.615874</td>
      </tr>
      <tr>
        <th>min</th>
        <td>-124.350000</td>
        <td>32.540000</td>
        <td>1.000000</td>
        <td>2.000000</td>
        <td>1.000000</td>
        <td>3.000000</td>
        <td>1.000000</td>
        <td>0.499900</td>
        <td>14999.000000</td>
      </tr>
      <tr>
        <th>25%</th>
        <td>-121.800000</td>
        <td>33.930000</td>
        <td>18.000000</td>
        <td>1447.750000</td>
        <td>296.000000</td>
        <td>787.000000</td>
        <td>280.000000</td>
        <td>2.563400</td>
        <td>119600.000000</td>
      </tr>
      <tr>
        <th>50%</th>
        <td>-118.490000</td>
        <td>34.260000</td>
        <td>29.000000</td>
        <td>2127.000000</td>
        <td>435.000000</td>
        <td>1166.000000</td>
        <td>409.000000</td>
        <td>3.534800</td>
        <td>179700.000000</td>
      </tr>
      <tr>
        <th>75%</th>
        <td>-118.010000</td>
        <td>37.710000</td>
        <td>37.000000</td>
        <td>3148.000000</td>
        <td>647.000000</td>
        <td>1725.000000</td>
        <td>605.000000</td>
        <td>4.743250</td>
        <td>264725.000000</td>
      </tr>
      <tr>
        <th>max</th>
        <td>-114.310000</td>
        <td>41.950000</td>
        <td>52.000000</td>
        <td>39320.000000</td>
        <td>6445.000000</td>
        <td>35682.000000</td>
        <td>6082.000000</td>
        <td>15.000100</td>
        <td>500001.000000</td>
      </tr>
    </tbody>
  </table>
</div>

Graficaremos un histograma de cada columna

```python
housing.hist(bins=50, figsize=(20,15))
plt.show()
```

![png](/images/end-to-end/end-to-end-1.png)


Se pueden observar varias cosas en estos histogramas
1. El ingreso medio no parece expresado en dolares. Los datos fueron escalados y limitados a 15 para grandes ingresos medios, y a 5 para los ingresos medios más bajos. Los números representan decenas de miles de dolares.
2. La edad media de la vivienda y el valor medio de la vivienda también fueron limitados.
3. Los atributos tienen escalas muy diferentes.
4. Varios histogramas tienen una larga cola: Se extienden más a la derecha de la mediana que a la izquierda.

## Crear conjunto de pruebas

Segmentaremos y clasificaremos la columna "median_income"
en 5 contenedores. De esta forma crearemos una columna categórica
que nos servirá para separar los datos equitativamente. 

```python
housing["income_cat"] = pd.cut(housing["median_income"],
                               bins=[0., 1.5, 3.0, 4.5, 6., np.inf],
                               labels=[1, 2, 3, 4, 5])
```

```python
housing["income_cat"].value_counts()
```

|3|7236|
|2|6581|
|4|3639|
|5|2362|
|1| 822|
|**Name**: income_cat| **dtype**: int64|

Ahora crearemos separaremos el conjunto de pruebas usando la nueva
columna categórica.

```python
from sklearn.model_selection import StratifiedShuffleSplit

split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_index, test_index in split.split(housing, housing["income_cat"]):
    strat_train_set = housing.loc[train_index]
    strat_test_set = housing.loc[test_index]
```

Comprobaremos que ambos conjuntos tienen la misma proporción de 
ingresos medios.

```python
strat_test_set["income_cat"].value_counts() / len(strat_test_set)
```

|3|    0.350533|
|2|    0.318798|
|4|    0.176357|
|5|    0.114583|
|1|    0.039729|
|Name: income_cat| dtype: float64|

```python
housing["income_cat"].value_counts() / len(housing)
```

|3|0.350581|
|2|0.318847|
|4|0.176308|
|5|0.114438|
|1|0.039826|
|Name: income_cat| dtype: float64|

Podemos observar que ambos conjuntos tienen una proporción similar.

Procederemos a eliminar la columna categórica que creamos.

```python
for set_ in (strat_train_set, strat_test_set):
    set_.drop("income_cat", axis=1, inplace=True)
```

## Visualizar los datos


```python
housing = strat_train_set.copy()
```

```python
housing.plot(kind="scatter", x="longitude", y="latitude", alpha=0.4,
             s=housing["population"]/100, label="population", figsize=(10,7),
             c="median_house_value", cmap=plt.get_cmap("jet"), colorbar=True,
)
plt.legend();
```

![png](/images/end-to-end/end-to-end-2.png)

Podemos apreciar que las casas con un mayor precio están ubicadas cerca
de la costa

## Buscar correlaciones

```python
corr_matrix = housing.corr()
```

```python
corr_matrix["median_house_value"].sort_values(ascending=False)
```
El coeficiente de correlación se distribuye de -1 a 1, si el valor es
cercano a -1 significa que dos columnas tienen una correlación negativa,
si es cercano a 1 significa que dos columnas tienen una correlación
positiva, si el valor es 0 significa que no existe una correlación entre
las dos columnas.

|median_house_value | 1.000000|
|median_income      | 0.687160|
|total_rooms        | 0.135097|
|housing_median_age | 0.114110|
|households         | 0.064506|
|total_bedrooms     | 0.047689|
|population         |-0.026920|
|longitude          |-0.047432|
|latitude           |-0.142724|
|Name: median_house_value| dtype: float64|

Ya que "median_house_value" es la columna que estamos evaluando da 1
exacto, la columna que tiene una mayor correlación es "median_income"
lo que nos dice que las casas con un mayor precio en un vecindario
son aquellas cuyos residentes tienen un ingreso medio mayor.

Graficaremos las 3 columnas con una mayor correlación, para intentar
apreciarlo mejor

```python
from pandas.plotting import scatter_matrix

attributes = ["median_house_value", "median_income", "total_rooms",
              "housing_median_age"]
scatter_matrix(housing[attributes], figsize=(12, 8));
```


![png](/images/end-to-end/end-to-end-3.png)

Ahora nos centraremos a la gráfica con una mayor correlación 
"median_house_value"

```python
housing.plot(kind="scatter", x="median_income", y="median_house_value",
             alpha=0.1);
```

![png](/images/end-to-end/end-to-end-4.png)


Esta gráfica revela unas cuantas cosas. Primero, la correlación es muy
fuerte; puedes ver claramente la tendencia y los puntos no están muy
dispersos. Segundo, el límite de precio es claramente visible como
una línea horizontal a los \$500,000, pero también se revelan otras
líneas menos obvias: una línea horizontal alrededor de \$450,000,
otra alrededor de \$350,000, y tal vez una alrededor de \$280,000
y algunas debajo de eso.

## Experimentar con combinaciones de atributos

Crearemos nuevas columnas para intentar buscar algunas con mayor 
correlación

```python
housing["rooms_per_household"] = housing["total_rooms"]/housing["households"]
housing["bedrooms_per_room"] = housing["total_bedrooms"]/housing["total_rooms"]
housing["population_per_household"]=housing["population"]/housing["households"]
```


```python
corr_matrix = housing.corr()
corr_matrix["median_house_value"].sort_values(ascending=False)
```

|median_house_value      | 1.000000|
|median_income           | 0.687160|
|rooms_per_household     | 0.146285|
|total_rooms             | 0.135097|
|housing_median_age      | 0.114110|
|households              | 0.064506|
|total_bedrooms          | 0.047689|
|population_per_household|-0.021985|
|population              |-0.026920|
|longitude               |-0.047432|
|latitude                |-0.142724|
|bedrooms_per_room       |-0.259984|
|Name: median_house_value| dtype: float64|



El nuevo atributo "bedrooms_per_room" está más correlacionado con
el valor medio de las viviendas que el número de habitaciones o
dormitorios. Aparentemente casas con un ratio dormitorios/habitaciones
más bajo tienden a ser más caras. El atributo "rooms_per_household"
es más informativo que el total de habitaciones en un distrito -
obviamente las casas más grandes son más caras

## Preparar los datos


```python
housing = strat_train_set.drop("median_house_value", axis=1)
housing_labels = strat_train_set["median_house_value"].copy()
```

### Limpieza de datos

Primero nos encargaremos de los valores nulos, rellenaremos esos valores
con la mediana de la columna 

```python
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(strategy="median")
```

Ya que el SimpleImputer solo puede trabajar con valores numéricos
retiraremos la columnas no numéricas

```python
housing_num = housing.drop("ocean_proximity", axis=1)
```

Ahora crearemos el imputer

```python
imputer.fit(housing_num)
```

Valores que usará el imputer para rellenar las columnas

```python
imputer.statistics_
```

|-118.51|34.26|29.0|2119.5|433.0|1164.0|408.0|3.5409|

Aplicaremos el imputer

```python
X = imputer.transform(housing_num)
```

```python
housing_tr = pd.DataFrame(X, columns=housing_num.columns)
```

Ahora nos encargaremos de las columnas categóricas, en este caso 
"ocean_proximity"

```python
housing_cat = housing[["ocean_proximity"]]
```

```python
housing_cat.head(10)
```
<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: left;">
      <th></th>
      <th>ocean_proximity</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>17606</th>
      <td>&lt;1H OCEAN</td>
    </tr>
    <tr>
      <th>18632</th>
      <td>&lt;1H OCEAN</td>
    </tr>
    <tr>
      <th>14650</th>
      <td>NEAR OCEAN</td>
    </tr>
    <tr>
      <th>3230</th>
      <td>INLAND</td>
    </tr>
    <tr>
      <th>3555</th>
      <td>&lt;1H OCEAN</td>
    </tr>
    <tr>
      <th>19480</th>
      <td>INLAND</td>
    </tr>
    <tr>
      <th>8879</th>
      <td>&lt;1H OCEAN</td>
    </tr>
    <tr>
      <th>13685</th>
      <td>INLAND</td>
    </tr>
    <tr>
      <th>4937</th>
      <td>&lt;1H OCEAN</td>
    </tr>
    <tr>
      <th>4861</th>
      <td>&lt;1H OCEAN</td>
    </tr>
  </tbody>
</table>
</div>

Usaremos el algoritmo OneHotEncoder para crear una columna por cada
nivel de la variable categórica, si una fila pertenece a un nivel
entonces el valor que tendrá en la columna será 1 en otro caso será
0

```python
from sklearn.preprocessing import OneHotEncoder
cat_encoder = OneHotEncoder()
housing_cat_1hot = cat_encoder.fit_transform(housing_cat)
```

Crearemos una clase que nos permita automatizar la inclusion de
la nueva columna que creamos anteriormente, gracias a esta clase
podremos darnos una idea si perjudica o ayuda al modelo.

```python
from sklearn.base import BaseEstimator, TransformerMixin

# column index
rooms_ix, bedrooms_ix, population_ix, households_ix = 3, 4, 5, 6

class CombinedAttributesAdder(BaseEstimator, TransformerMixin):
    def __init__(self, add_bedrooms_per_room = True): # no *args or **kargs
        self.add_bedrooms_per_room = add_bedrooms_per_room
    def fit(self, X, y=None):
        return self  # nothing else to do
    def transform(self, X):
        rooms_per_household = X[:, rooms_ix] / X[:, households_ix]
        population_per_household = X[:, population_ix] / X[:, households_ix]
        if self.add_bedrooms_per_room:
            bedrooms_per_room = X[:, bedrooms_ix] / X[:, rooms_ix]
            return np.c_[X, rooms_per_household, population_per_household,
                         bedrooms_per_room]
        else:
            return np.c_[X, rooms_per_household, population_per_household]

attr_adder = CombinedAttributesAdder(add_bedrooms_per_room=False)
housing_extra_attribs = attr_adder.transform(housing.values)
```
Le echaremos un vistazo a la nueva tabla

```python
housing_extra_attribs = pd.DataFrame(
    housing_extra_attribs,
    columns=list(housing.columns)+["rooms_per_household", "population_per_household"],
    index=housing.index)
housing_extra_attribs.head()
```

<div class="table-wrapper">
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>longitude</th>
      <th>latitude</th>
      <th>housing_median_age</th>
      <th>total_rooms</th>
      <th>total_bedrooms</th>
      <th>population</th>
      <th>households</th>
      <th>median_income</th>
      <th>ocean_proximity</th>
      <th>rooms_per_household</th>
      <th>population_per_household</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>17606</th>
      <td>-121.89</td>
      <td>37.29</td>
      <td>38</td>
      <td>1568</td>
      <td>351</td>
      <td>710</td>
      <td>339</td>
      <td>2.7042</td>
      <td>&lt;1H OCEAN</td>
      <td>4.62537</td>
      <td>2.0944</td>
    </tr>
    <tr>
      <th>18632</th>
      <td>-121.93</td>
      <td>37.05</td>
      <td>14</td>
      <td>679</td>
      <td>108</td>
      <td>306</td>
      <td>113</td>
      <td>6.4214</td>
      <td>&lt;1H OCEAN</td>
      <td>6.00885</td>
      <td>2.70796</td>
    </tr>
    <tr>
      <th>14650</th>
      <td>-117.2</td>
      <td>32.77</td>
      <td>31</td>
      <td>1952</td>
      <td>471</td>
      <td>936</td>
      <td>462</td>
      <td>2.8621</td>
      <td>NEAR OCEAN</td>
      <td>4.22511</td>
      <td>2.02597</td>
    </tr>
    <tr>
      <th>3230</th>
      <td>-119.61</td>
      <td>36.31</td>
      <td>25</td>
      <td>1847</td>
      <td>371</td>
      <td>1460</td>
      <td>353</td>
      <td>1.8839</td>
      <td>INLAND</td>
      <td>5.23229</td>
      <td>4.13598</td>
    </tr>
    <tr>
      <th>3555</th>
      <td>-118.59</td>
      <td>34.23</td>
      <td>17</td>
      <td>6592</td>
      <td>1525</td>
      <td>4459</td>
      <td>1463</td>
      <td>3.0347</td>
      <td>&lt;1H OCEAN</td>
      <td>4.50581</td>
      <td>3.04785</td>
    </tr>
  </tbody>
</table>
</div>

Crearemos una pipeline que nos permita hacer todo el pre-procesamiento
anterior de una forma más cómoda, primero nos enfocaremos a las columnas
numéricas, aprovecharemos para estadarizar los valores de las columnas.

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

num_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy="median")),
        ('attribs_adder', CombinedAttributesAdder()),
        ('std_scaler', StandardScaler()),
    ])

housing_num_tr = num_pipeline.fit_transform(housing_num)
```

Ahora crearemos una pipeline para todo el conjunto de datos

```python
from sklearn.compose import ColumnTransformer

num_attribs = list(housing_num)
cat_attribs = ["ocean_proximity"]

full_pipeline = ColumnTransformer([
        ("num", num_pipeline, num_attribs),
        ("cat", OneHotEncoder(), cat_attribs),
    ])

housing_prepared = full_pipeline.fit_transform(housing)
```

## Seleccionar un modelo y entrenarlo

### Entrenamiento y evaluación en el conjunto de entrenamiento

Probaremos varios modelos para encontrar el que tenga mejores resultados

Primero intentaremos con un modelo de regresión lineal

```python
from sklearn.linear_model import LinearRegression

lin_reg = LinearRegression()
lin_reg.fit(housing_prepared, housing_labels)
```
Intentaremos hacer unas cuantas predicciones con este modelo

```python
# let's try the full preprocessing pipeline on a few training instances
some_data = housing.iloc[:5]
some_labels = housing_labels.iloc[:5]
some_data_prepared = full_pipeline.transform(some_data)

print("Predictions:", lin_reg.predict(some_data_prepared))
print("Labels:", list(some_labels))
```

|Predictions|210644.60|317768.80|210956.43|59218.98|189747.55|
|Labels     |286600.00|340600.00|196900.00|46300.00|254500.00|

Ahora que tenemos tanto los valores predichos como los valores reales
usaremos algunas métricas para evaluar el rendimiento del modelo

Primero provaremos el error medio cuadrado

```python
from sklearn.metrics import mean_squared_error

housing_predictions = lin_reg.predict(housing_prepared)
lin_mse = mean_squared_error(housing_labels, housing_predictions)
lin_rmse = np.sqrt(lin_mse)
```

|lin_rmse|68628.19|

Ahora probaremos con el error medio absoluto

```python
from sklearn.metrics import mean_absolute_error

lin_mae = mean_absolute_error(housing_labels, housing_predictions)
```

|lin_mae|49439.89|

Ahora probaremos un árbol de decisión 

```python
from sklearn.tree import DecisionTreeRegressor

tree_reg = DecisionTreeRegressor(random_state=42)
tree_reg.fit(housing_prepared, housing_labels)
```
Ya que tenemos el modelo creado lo usaremos para hacer predicciones
y verificar su rendimiento

```python
housing_predictions = tree_reg.predict(housing_prepared)
tree_mse = mean_squared_error(housing_labels, housing_predictions)
tree_rmse = np.sqrt(tree_mse)
```

|tree_rmse|0.0|

El cuadrado medio nos dio un resultado 0, lo cual significa que el modelo
acertó todas las predicciones, sin embargo esto puede implicar que el 
modelo se haya sobreajustado, lo que nos dice que aunque haya acertado
en el conjunto de entrenamiento es posible que falle con nueva información

### Evaluación usando Cross validation

Ya que el modelo parece que sobreajustó las predicciones usaremos un
algoritmo que divide el conjunto en cierto número de partes iguales,
en nuestro caso lo dividiremos en 10 partes, entonces entrena el modelo
con 9 de las partes y hace las predicciones con la parte restante, esto 
la hará 10 veces, tomando diferentes partes cada vez.

```python
from sklearn.model_selection import cross_val_score

scores = cross_val_score(tree_reg, housing_prepared, housing_labels,
                         scoring="neg_mean_squared_error", cv=10)
tree_rmse_scores = np.sqrt(-scores)
```

Ya que tenemos calculados los valores los imprimiremos

```python
def display_scores(scores):
    print("Scores:", scores)
    print("Mean:", scores.mean())
    print("Standard deviation:", scores.std())

display_scores(tree_rmse_scores)
```

|Scores  |70194.33|66855.16|72432.58|70758.73|71115.88|
|        |75585.14|70262.86|70273.63|75366.87|71231.65|
|Mean    |71407.68|
|Standard deviation|2439.43|

Haremos lo mismo con la regresión lineal

```python
lin_scores = cross_val_score(lin_reg, housing_prepared, housing_labels,
                             scoring="neg_mean_squared_error", cv=10)
lin_rmse_scores = np.sqrt(-lin_scores)
display_scores(lin_rmse_scores)
```

|Scores|66782.73|66960.11|70347.95|74739.57|68031.13|
|      |71193.84|64969.63|68281.61|71552.91|67665.10|
|Mean  |69052.46|
|Standard deviation|2731.67|

Ahora probaremos un bosque aleatorio

```python
from sklearn.ensemble import RandomForestRegressor

forest_reg = RandomForestRegressor(n_estimators=100, random_state=42)
forest_reg.fit(housing_prepared, housing_labels)
```

```python
housing_predictions = forest_reg.predict(housing_prepared)
forest_mse = mean_squared_error(housing_labels, housing_predictions)
forest_rmse = np.sqrt(forest_mse)
```

|forest_rmse|18603.515021376355|


```python
from sklearn.model_selection import cross_val_score

forest_scores = cross_val_score(forest_reg, housing_prepared, housing_labels,
                                scoring="neg_mean_squared_error", cv=10)
forest_rmse_scores = np.sqrt(-forest_scores)
display_scores(forest_rmse_scores)
```

|Scores|49519.80|47461.91|50029.02|52325.28|49308.39|
|      |53446.37|48634.80|47585.73|53490.10|50021.58|
|Mean  |50182.30|
|Standard deviation|2097.08|

Parece ser que el modelo que mejor rindió fue el del bosque aleatorio

Ahora que tenemos nuestro modelo lo afinaremos para que de mejores
resultados, para eso usaremos GridSearchCV para encontrar el valor de los hyper-parametros
más adecuado

```python
from sklearn.model_selection import GridSearchCV

param_grid = [
    # try 12 (3×4) combinations of hyperparameters
    {'n_estimators': [3, 10, 30], 'max_features': [2, 4, 6, 8]},
    # then try 6 (2×3) combinations with bootstrap set as False
    {'bootstrap': [False], 'n_estimators': [3, 10], 'max_features': [2, 3, 4]},
  ]

forest_reg = RandomForestRegressor(random_state=42)
# train across 5 folds, that's a total of (12+6)*5=90 rounds of training 
grid_search = GridSearchCV(forest_reg, param_grid, cv=5,
                           scoring='neg_mean_squared_error',
                           return_train_score=True)
grid_search.fit(housing_prepared, housing_labels)
```

Ahora veremos cual es la mejor combinación de hyper-parametros 

```python
grid_search.best_params_
```

|max_features|8   |
|n_estimators|30  |
|bootstrap   |True|

Calcularemos el error medio cuadrado para cada combinación de hyper-parametros
para verificar que la combinación sea correcta

```python
cvres = grid_search.cv_results_
for mean_score, params in zip(cvres["mean_test_score"], cvres["params"]):
    print(np.sqrt(-mean_score), params)
```

|mean_squared_error|bootstrap|max_features|n_estimators|
|:----------------:|:-------:|:----------:|:----------:|
|63669.11          |True     |2           |3           |
|55627.09          |True     |2           |10          |
|53384.57          |True     |2           |30          |
|60965.95          |True     |4           |3           |
|52741.04          |True     |4           |10          |
|50377.40          |True     |4           |30          |
|58663.93          |True     |6           |3           |
|52006.19          |True     |6           |10          |
|50146.51          |True     |6           |30          |
|57869.25          |True     |8           |3           |
|51711.12          |True     |8           |10          |
|49682.27          |True     |8           |30          |
|62895.06          |False    |2           |3           |
|54658.17          |False    |2           |10          |
|59470.40          |False    |3           |3           |
|52724.98          |False    |3           |10          |
|57490.56          |False    |4           |3           |
|51009.49          |False    |4           |10          |

Ahora buscaremos las columnas que aportan más al modelo, en otras palabras,
las columnas que afectan más al precio de una vivienda

```python
feature_importances = grid_search.best_estimator_.feature_importances_
feature_importances
```

|feature_importances|7.334e-02|6.290e-02|4.114e-02|1.467e-02|
|                   |1.410e-02|1.487e-02|1.425e-02|3.661e-01|
|                   |5.641e-02|1.087e-01|5.335e-02|1.031e-02|
|                   |1.647e-01|6.028e-05|1.960e-03|2.856e-03|

Obtuvimos lo que afecta cada columna en el precio, sin embargo nos falta
anotar cada columna con su valor.

```python
extra_attribs = ["rooms_per_hhold", "pop_per_hhold", "bedrooms_per_room"]
#cat_encoder = cat_pipeline.named_steps["cat_encoder"] # old solution
cat_encoder = full_pipeline.named_transformers_["cat"]
cat_one_hot_attribs = list(cat_encoder.categories_[0])
attributes = num_attribs + extra_attribs + cat_one_hot_attribs
sorted(zip(feature_importances, attributes), reverse=True)
```

|0.3661    |median_income     |
|0.1647    |INLAND            |
|0.1087    |pop_per_hhold     |
|0.0733    |longitude         |
|0.0629    |latitude          |
|0.0564    |rooms_per_hhold   |
|0.0533    |bedrooms_per_room |
|0.0411    |housing_median_age|
|0.0148    |population        |
|0.0146    |total_rooms       |
|0.0142    |households        |
|0.0141    |total_bedrooms    |
|0.0103    |<1H OCEAN         |
|0.0028    |NEAR OCEAN        |
|0.0019    |NEAR BAY          |
|6.0280e-05|ISLAND            |

Ahora probaremos nuestro modelo con el conjunto de pruebas que separamos en 
un principio

```python
final_model = grid_search.best_estimator_

X_test = strat_test_set.drop("median_house_value", axis=1)
y_test = strat_test_set["median_house_value"].copy()

X_test_prepared = full_pipeline.transform(X_test)
final_predictions = final_model.predict(X_test_prepared)

final_mse = mean_squared_error(y_test, final_predictions)
final_rmse = np.sqrt(final_mse)
```

|final_rmse|47730.22|

El resultado final nos dice que nuestro modelo puede predecir el valor de una 
casa en California con un margen de error de $47,730.22