# README #

# El Efecto del Pre-procesamiento de Textos en Clasificadores del Lenguaje Natural
## Casos de uso en análisis de sentimientos y detección de sátiras en noticias

En este repositorio se encuentran los resultados de todos los experimentos ejecutados con el objetivo de evaluar los efectos positivos y negativos del pre-procesamiento en la clasificación automática de textos, aplicando técnicas de limpieza y ponderación tradicionales desde el punto de vista de la representación de los mismos usando el modelo de espacio vectorial en el contexto de documentos formales e informales.


## Para empezar
En la carpeta de experimentos se encuentran los archivos de Jupyter, entorno sobre el cual se ejecutaron los algoritmos (es un entorno web para ejecutar código en Python). Estos archivos son llamados notebooks y estan separados por clasificador. Par los casos en que se aplica el clasificador SVM radial, también están separados por metodo de ponderación, ya que eran los que más tardaban en ejecutarse, entonces se dividieron así por agilidad en la ejecución.

Cada archivo esta nombrado de la siguiente forma:

idioma_dominio_clasificador.ipynb

Ejemplo:

es_twitter_sa_svm_lineal.ipynb

Los dominios son:

* twitter_sa = Análisis de sentimientos en tweets
* news_fakes = Detección de sátiras en noticias


## Construido con
* Los experimentos se ejecutan con scikit-learn y NLTK. En cada notebook se pueden ver los paquetes usados de cada framework.

* Los conjuntos no se pueden compartir por políticas de Twitter pero pueden ser descargados a través del API (leer documento del trabajo de grado).


## Información
Para información adicional o instrucciones detalladas por favor contacte a:

* Camilo Torres Botero (ctorres9@eafit.edu.co)


## Trabajo asesorado por
* Marta Silvia Tabares Betancur (mtabares@eafit.edu.co)

