---
layout: post
title:  "Cubo de Rubik"
date:   2021-01-03 13:21:00 -0600
category: ProblemSolver
---

Modelo de un cubo de Rubik Interactivo que puede resolverse sin ayuda de un ser humano,
utilizando un algoritmo A\* de profundidad iterativa

<div class="row">
  <div class="col">
    <img src="\images\rubik\python_logo.png" alt="python logo" width="120"/>
  </div>
  <!--more-->
  <div class="col">
    <img src="\images\rubik\pygame_logo.png" alt="python logo" width="300"/>
  </div>
</div>

El cubo de Rubik es una figura tridimensional que fue pensado como un puzzle el cual se pueden
girar sus caras de tal forma que cada giro cambie el orden de los colores que contiene cada una
de sus caras, el estado solucionado está definido como un cubo el cual tiene un sólo color en cada
una de sus caras.

El puzzle en cuestión está compuesto por 26 cuboides más pequeños, existen 3 variaciones de estos
cuboides:

- **Centro**: Su función principal es fungir como eje de simetría, de tal forma que al momento de girar
cada una cara los centros mantendrán su posición, el cubo contiene 6 piezas de este tipo.

- **Arista**: Esta pieza se encuentra entre dos centros diferentes, por lo que contiene dos colores diferentes.

- **Esquina**: Esta pieza contiene 3 colores diferentes. cada esquina se encuentra entre tres aristas, 
lo que la convierte en la pieza con más orientaciones diferentes.

Cada giro del cubo de Rubik es un grupo cíclico de rango 4, lo que nos dice que al hacer
4 veces el mismo giro, el cubo retomará la forma que tenía antes de hacer el primero de los 4 giros.

El cubo tiene un total de **43 252 003 274 489 856 000** combinaciones posibles lo que hace que resolverlo
por movimientos al azar sea prácticamente imposible.

## Modelado del cubo de Rubik

### Colores del cubo

Para representar los colores utilicé tri-tuplas que representan los colores en formato RGB

```python
  GREEN  = (0,   155, 72)
  BLACK  = (0,   0,   0)
  WHITE  = (255, 255, 255)
  RED    = (183, 18,  52)
  YELLOW = (255, 213, 0)
  BLUE   = (0,   70,  173)
  ORANGE = (255, 88,  0)
```

### Modelado de las caras del cubo

Para modelar las caras utilicé listas que contienen el color de cada una de las piezas que contienen
esa cara, sin contar la pieza central ya que esta no cambia de posición.
El cubo se genera con los colores del cubo resuelto.


```python
  class Rubik_cube:
    def __init__(self):
      self.up = [WHITE]*8
      self.left = [RED]*8
      self.center = [BLUE]*8
      self.right = [ORANGE]*8
      self.back = [GREEN]*8
      self.down = [YELLOW]*8
```

### Modelado de los giros del cubo

Cada cara puede hacer 3 tipos de giros, sentido horario, sentido anti-horario y un giro de 180º,
ya que los giros están formados por permutaciones, entonces se creó un método que permite hacer
las permutaciones entre una lista de elementos.

Si al método se le da el parámetro "face" esto quiere decir, que todas las permutaciones se darán en
la misma cara, en dicho caso sólo se le dará la posición que intercambiará las piezas.

En caso contrarío se le pasarán una serie de argumentos en forma de tupla que equivaldrían a la cara 
y la posición que se hará cada permutación

```python
    def permutate(self, *args, **kwargs):
        if "face" in kwargs:
            face = kwargs["face"]
            face[args[0]], face[args[1]], face[args[2]], face[args[3]] =\
                face[args[3]], face[args[0]], face[args[1]], face[args[2]]
        else:
            args[0][0][args[0][1]], args[1][0][args[1][1]], args[2][0][args[2][1]], args[3][0][args[3][1]] =\
                args[3][0][args[3][1]], args[0][0][args[0][1]], args[1][0][args[1][1]], args[2][0][args[2][1]]
```

Los tres tipos de giros contendrán la misma lógica, le enviarán a las piezas al método de permutación
y este se encargará de hacer los giros.
Cada movimiento afecta a 20 colores diferentes, pero al ser permutaciones de grado 4, eso significa
que cada giro hace 5 permutaciones simultaneas, de las cuales 2 pertenecen a la misma cara y las 3 
restantes pertenecen a las caras adyacentes, dos por las esquinas y otra por el centro.

```python
  def move_up_prime(self):
    # Giro de la cara
    self.permutate(0, 2, 7, 5, face=self.up)
    self.permutate(1, 4, 6, 3, face=self.up)

    # Giro del centro
    self.permutate((self.back, 1), (self.right, 1), 
      (self.center, 1), (self.left, 1)) 

    # Giro de las esquinas
    self.permutate((self.back, 0), (self.right, 0), 
      (self.center, 0), (self.left, 0))        
    self.permutate((self.back, 2), (self.right, 2), 
      (self.center, 2), (self.left, 2))
```

Para realizar el giro en sentido antihorario se envían los datos, pero de forma inversa.

En caso de la versión de 180º, se ejecuta dos veces la versión de giro horario.

```python
  def move_up_180(self):
    self.move_up()
    self.move_up()
```
### Deshacer el cubo

Para deshacer el cubo se harán giros de forma aleatoria de tal forma que no se gire la misma
cara dos veces seguidas y en caso de que se gire una cara y su cara opuesta de forma consecutiva,
no se podrán girar ninguna de las dos hasta que una tercera cara sea girada. De esta forma se evitará
que el cubo genere estados que no estén completamente deshechos.

Ya que el número máximo de movimientos que se requieren para la solución de cualquier estado del 
cubo de Rubik son 20, sólo se harán 20 movimientos para deshacer el cubo.

Los movimientos se guardarán en una lista de listas que incluyen cada uno de los movimientos posibles.

Las caras opuestas están definidas en un diccionario, cuyas claves y valores están especificados
con el índice de los movimientos.

El resultado se enviará en forma de lista de métodos, de esta forma se podrá ejecutar en otras partes
del código.

```python
  def shuffle(self):
    complements = {0:3, 1:4, 2:5, 3:0, 4:1, 5:2}
    counter = 0
    result = []
    moves = [[self.move_up,     self.move_up_180,     self.move_up_prime],
             [self.move_left,   self.move_left_180,   self.move_left_prime],
             [self.move_center, self.move_center_180, self.move_center_prime],
             [self.move_down,   self.move_down_180,   self.move_down_prime],
             [self.move_right,  self.move_right_180,  self.move_right_prime],
             [self.move_back,   self.move_back_180,   self.move_back_prime]]
    first = None
    second = None
    while(counter < 20):
      choices = random.choice(moves)
      if choices == first or choices == second:
        continue
      if first == None:
        first = choices
      else:
        index = moves.index(choices)
        if complements[index]==moves.index(first):
          second = choices
        else:
          first = choices
          second = None
      choice = random.choice(choices)
      result.append(choice)
      counter += 1
    return result
```
### Representación del objeto del cubo de Rubik

Para que la información que contiene el cubo de Rubik pueda ser utilizada en otras partes del código,
se necesita hacer una representación del objeto, en este caso se cambiarán los valores que contienen
las caras, por las iniciales del color que los representa, separando cada cara con un espacio.

Para cambiar la tupla por un string se utilizará un diccionario que contiene como claves las tuplas
que representan los colores y sus valores será la primera letra de cada color.

```python
colors = {GREEN:"G", WHITE:"W", RED: "R", 
          YELLOW:"Y", BLUE:"B", ORANGE: "O"}
```

Ese diccionario se utilizará para crear el string que contiene la representación del cubo de Rubik

```python
  def __repr__(self):
    result = ""
    for color in self.up:
      result += colors[color]
      result += " "
    for color in self.left:
      result += colors[color]
      result += " "
    for color in self.center:
      result += colors[color]
      result += " "
    for color in self.right:
      result += colors[color]
      result += " "
    for color in self.back:
      result += colors[color]
      result += " "
    for color in self.down:
      result += colors[color]
      result += " "
    return result
```

### Obtener la puntuación del cubo

Para el desarrollo de la heurística, se necesita puntuar el estado actual del cubo, de esta forma
se sabrá si un movimiento mejorará el puntaje del cubo o lo empeorará, de esta forma, el algoritmo
no tendrá que revisar todas las combinaciones posibles, sino que solo revisará las mejores opciones
disponibles.

Para determinar el puntaje se tomó en cuenta cada color que contiene las caras, si el color pertenece 
a la cara, entonces no se sumará nada, si el color pertenece a la cara opuesta, entonces se sumarán dos
puntos y en dado caso de que el color pertenezca a alguna de las caras adyacentes se sumará un punto.
El algoritmo decidirá el movimiento que dé el puntaje con menor valor.

Un cubo resuelto dará una puntuación de 0.

```python
  def getScore(self):
    i = 0
    count = 0
    colors = ["W", "R", "B", "O", "G", "Y"]
    for c in repr(self):
      if c == " ":
        i += 1
        continue
      if c == colors[i]:
        count += 0
      elif c == complements[colors[i]]:
        count += 2
      else:
        count += 1
    return count
```

### Obtener una copia del cubo

Durante la ejecución del programa se necesitarán crear copias del cubo original, ya que tendremos
que probar las diferentes posibilidades y verificar la que nos otorgue más posibilidades de lograr
nuestro objetivo.

```python
  def getCopy(self):
    copy = Rubik_cube()
    for i in range(8):
      copy.up[i] = self.up[i]
      copy.left[i] = self.left[i]
      copy.center[i] = self.center[i]
      copy.right[i] = self.right[i]
      copy.back[i] = self.back[i]
      copy.down[i] = self.down[i]
    return copy
```

## Resolver el cubo de Rubik

Para la resolución del cubo se creará una lista de strings que contienen cada uno de los giros 
disponibles, pero los métodos estarán guardados con respecto a la copia, en lugar del cubo original.
El nodo actual será una copia del cubo de Rubik original, y las copias en las que se evaluarán las
posibilidades son copias del cubo del nodo actual.

En caso de que el cubo esté resuelto desde el primer momento se enviará un print, para evitar que
cause un error al no enviar ningún paso.

Para obtener el mejor movimiento, ejecutaremos cada uno de los movimientos posibles y guardaremos
su calificación en un diccionario. Con el cual podremos verificar cual de ellos es la mejor opción.

Cuando conozcamos cuál es el mejor de los movimientos tendremos que actualizar el nodo actual,
por lo cual cambiaremos el método en forma de string de "copy" a "current", y lo ejecutaremos, 
de esta forma actualizaremos el nodo actual. Agregaremos la representación del cubo a una lista para
evitar que los estados se repitan.

Por último agregaremos, el método a la lista del resultado, cambiándolo para que este referenciando
al cubo original, en lugar de una copia.

```python
def solve(cube):
  path = []
  current = cube.getCopy()
  currentScore = cube.getScore()
  if(currentScore==0):
    return [print]
  result = []
  moves = [["copy.move_up()",     "copy.move_up_180()",     "copy.move_up_prime()"],
           ["copy.move_left()",   "copy.move_left_180()",   "copy.move_left_prime()"],
           ["copy.move_center()", "copy.move_center_180()", "copy.move_center_prime()"],
           ["copy.move_down()",   "copy.move_down_180()",   "copy.move_down_prime()"],
           ["copy.move_right()",  "copy.move_right_180()",  "copy.move_right_prime()"],
           ["copy.move_back()",   "copy.move_back_180()",   "copy.move_back_prime()"]]
  count = 0
  while(count < 100 and current.getScore()!=0):
    scores = {}
    path.append(str(current))
    count += 1
    for face in moves:
      for move in face:
        copy = current.getCopy()
        exec(move)
        if(str(copy) in path):
          continue
        scores[move] = copy.getScore()
    best = min(scores, key=scores.get)
    update = best[:1] + "urrent" + best[4:]
    exec(update)
    moving = "result.append("+best[:1] + "ube" + best[4:-2]+")"
    exec(moving)

  return result
```

## Entorno gráfico del cubo de Rubik

Para la representación gráfica del cubo de Rubik utilicé la librería pygame de python, la cual me 
permite dibujar gráficos en 2D.

La pantalla tiene un tamaño de 1000x600, cada pieza del cubo de Rubik tendrá un tamaño de 25x25,
los botones para interactuar con el cubo están agrupados en 3 columnas y 6 filas,
las columnas corresponden a los giros antihorario, 180º y horario respectivamente, y cada color
corresponde a la cara que se girará.

Adicionalmente hay dos botones, uno que permitirá deshacer completamente el cubo y otro que llamará
al algoritmo de resolución.

Dependiendo de la coordenada en la que se realice el clic, ejecutará una acción u otra.

Los métodos de deshacer y resolver retornan una lista de movimientos para que al momento de ejecutar
los movimientos se puedan apreciar los giros que se hacen y no sólo el resultado final.

Así que se llama a un método que ejecutará cada método de la lista, actualizará la pantalla en las
coordenadas del cubo y hará una pausa para que se pueda apreciar el movimiento.

```python
def execute(methods):
    for method in methods:
      method()
      paint_cube()
      pygame.display.update((300, 0, 600, 600))
      time.sleep(0.25)
```

### Resultado final

#### Estado final
![Cubo resuelto](\images\rubik\Solved.png)

#### Estado deshecho
![Cubo deshecho](\images\rubik\Unsolved.png)

#### Deshacer cubo
![Deshacer cubo](\images\rubik\mix.gif)

#### Armar cubo
![Armar cubo](\images\rubik\solving.gif)

