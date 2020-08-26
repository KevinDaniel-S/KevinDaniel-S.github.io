---
layout: post
title:  "Índice mundial de la felicidad"
date:   2020-08-26 14:19:00 -0500
category: Machine_Learning
---

El índice global de felicidad es una publicación anual de la Red de Soluciones
para el Desarrollo Sostenible de las Naciones Unidas (UN SDSN). Contiene articulos
y rankings de felicidad nacional basado en las respuestas de personas encuestadas
sobre sus vidas. Cada reporte se publicó en Marzo del año en cuestión.

<!--more-->

## Obtención de los datos

El conjunto de datos fue obtenido de [Kaggle world happiness report](https://www.kaggle.com/mathurinache/world-happiness-report)

## Limpieza

El conjunto de datos estaba dividido en 6 archivos diferentes uno para
año, desde el 2015 al 2020. Se cambiarán los nombres de las columnas
para que concuerden con respecto al resto. También se eliminarán las
columnas que no estén presentes en los demás archivos. Las regiones de
algunas tablas tienen diferentes nombres, por lo que se utilizará un
diccionario para darles un valor concordante.

Una vez que los 6 archivos estén unidos, cambiaremos los nombres de los 
países que están escritos de forma diferente en cada tabla. La esperanza
de vida del reporte del 2020 está en un formato de 0-100 mientras
que en el resto de reportes está en un formato de 0-1, por lo que se
modificará. Para rellenar los datos vacíos se utilizará la media de los
datos agrupados por país, si no hay datos sobre ese país, entonces 
se utilizará la media de la región.

Ahora miremos los estadísticos descriptivos de los datos

<div class="table-wrapper">

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Happiness Rank</th>
      <th>Happiness Score</th>
      <th>Economy (GDP per Capita)</th>
      <th>Family</th>
      <th>Health (Life Expectancy)</th>
      <th>Freedom</th>
      <th>Trust (Government Corruption)</th>
      <th>Generosity</th>
      <th>Dystopia Residual</th>
      <th>Social support</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>935.000000</td>
      <td>935.000000</td>
      <td>935.000000</td>
      <td>935.000000</td>
      <td>935.000000</td>
      <td>935.000000</td>
      <td>935.000000</td>
      <td>935.000000</td>
      <td>935.000000</td>
      <td>935.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>78.420321</td>
      <td>5.394436</td>
      <td>0.908311</td>
      <td>0.989542</td>
      <td>0.617658</td>
      <td>0.472008</td>
      <td>0.148801</td>
      <td>0.180425</td>
      <td>2.060878</td>
      <td>1.080953</td>
    </tr>
    <tr>
      <th>std</th>
      <td>45.021905</td>
      <td>1.124935</td>
      <td>0.402023</td>
      <td>0.297888</td>
      <td>0.229147</td>
      <td>0.201962</td>
      <td>0.130846</td>
      <td>0.153977</td>
      <td>0.539708</td>
      <td>0.279657</td>
    </tr>
    <tr>
      <th>min</th>
      <td>1.000000</td>
      <td>2.566900</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>-0.300907</td>
      <td>0.257241</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>39.500000</td>
      <td>4.540000</td>
      <td>0.600264</td>
      <td>0.812920</td>
      <td>0.500955</td>
      <td>0.337772</td>
      <td>0.061079</td>
      <td>0.098152</td>
      <td>1.739470</td>
      <td>0.874162</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>78.000000</td>
      <td>5.353500</td>
      <td>0.974380</td>
      <td>1.032809</td>
      <td>0.653133</td>
      <td>0.465820</td>
      <td>0.106285</td>
      <td>0.183000</td>
      <td>2.071238</td>
      <td>1.105000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>117.000000</td>
      <td>6.198500</td>
      <td>1.228785</td>
      <td>1.221453</td>
      <td>0.779015</td>
      <td>0.585785</td>
      <td>0.187788</td>
      <td>0.262000</td>
      <td>2.399977</td>
      <td>1.298576</td>
    </tr>
    <tr>
      <th>max</th>
      <td>158.000000</td>
      <td>7.808700</td>
      <td>2.096000</td>
      <td>1.610574</td>
      <td>1.141000</td>
      <td>0.974998</td>
      <td>0.890216</td>
      <td>0.838075</td>
      <td>3.837720</td>
      <td>1.644000</td>
    </tr>
  </tbody>
</table>
</div>


## Análisis exploratorio

### Confianza en el gobierno

La información de la  columna de confianza en el gobierno fue 
recopilada por la pregunta:
¿La corrupción está distribuida en todo el gobierno o no?

![png](\images\felicidad\output_61_0.png)

Podemos apreciar que en la región "Australia y nueva Zelanda" es donde
los resultados apuntaron a una percepción de corrupción menor,
mientras que en "Europa Central y del este" fueron los más altos seguido
muy de cerca por "Lationamerica y el Caribe"

### Economía (PIB per cápita)

El PIB está en terminos de Paridad de poder adquisitivo (PPP) ajustado
al valor de los dolares internacionales del 2011, tomados de los
Indicadores de Desarrollo Mundial (WDI) publicado por el banco mundial.

![png](\images\felicidad\output_62_0.png)

Podemos apreciar que en la región "Norte America" es donde el PIB per 
cápita es más alto, seguido de "Europa Oriental" y "Australia y nueva
Zelanda". Mientras que "África subsahariana" tiene los niveles más bajos"

### Libertad de tomar decisiones

La información de esta columna fue obtenida por los resultados de la 
pregunta: ¿Estás satisfecho con tu libertad para escoger lo que
quieres con tu vida?

![png](\images\felicidad\output_65_0.png)

La percepción de libertad en "Australia y Nueva Zelanda" tienen los valores
más altos. Mientras que "África subsahariana" tiene los niveles más bajos.

### Generosidad

Es la regresión residual de las respuestas a la pregunta: ¿Has donado
dinero a la caridad en el último mes?

![png](\images\felicidad\output_66_0.png)

La generosidad en "Australia y Nueva Zelanda" es la mayor, mientras que 
en "Europa Central y del este" son los niveles más bajos

### Familia

Familia es el promedio nacional de la pregunta: Si 
estuvieras en problemas, ¿Tienes parientes con los que
puedas contar  para que te ayuden cuando los necesites?

![png](\images\felicidad\output_63_0.png)

### Soporte Social

El soporte social es el promedio nacional de la pregunta: Si 
estuvieras en problemas, ¿Tienes amigos con los que
puedas contar  para que te ayuden cuando los necesites?

![png](\images\felicidad\output_68_0.png)


### Esperanza de vida

![png](\images\felicidad\output_64_0.png)

### Distopía residual 

Distopía es un país hipotético, llamado así porque tiene los valores
iguales a los promedios nacionales más bajos del mundo, para cada una de 
las variables. Se usó Distopía como un punto de referencia para
comparar las contribuciones para cada uno de los factores.
Cada país fue comparado con este país ficticio.

![png](\images\felicidad\output_67_0.png)

## Felicidad


![png](\images\felicidad\output_53_0.png)

![png](\images\felicidad\output_54_0.png)

![png](\images\felicidad\output_55_0.png)

![png](\images\felicidad\output_56_0.png)

![png](\images\felicidad\output_58_0.png)

![png](\images\felicidad\output_59_0.png)


## México con respecto al mundo

Para obtener los resultados de esta gráficas se escalaron los valores
del 0 al 10 donde 0 sería en valor mínimo encontrado en cada columna
mientras que 10 sería el máximo encontrado, de esta forma los
resultados se apreciarían mejor al tener una escala constante.

![png](\images\felicidad\output_76_0.png)

![png](\images\felicidad\output_77_0.png)


## Correlación

Para calcular el siguiente mapa se utilizó Coeficiente de 
correlación de Pearson donde un valor igual a 1 significa que
las variables tienen una correlación positiva perfecta (Si una variable
crece, la otra también), pero si tiene un valor -1 implica que
las variables tienen una correlación negativa perfecta (Si una 
variable crece, la otra decrece), además si el valor es 0 nos
dice que no existe una correlación entre las variables.

![png](\images\felicidad\output_79_0.png)

Ahora mostramos solo las correlaciones de la columna felicidad con el resto
de variables.

![png](\images\felicidad\output_81_0.png)

