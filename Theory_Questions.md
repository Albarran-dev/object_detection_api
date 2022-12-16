# Apartado 2:

## 1. Pasos Necesarios a Seguir

Aún el caso de que **no** existan modelos preentrenados con las clases que necesitamos, estos modelos siguen siendo de gran utilidad, podemos usarlos como 'feature extractors' y reentrenar sólo las cabezas del mismo(**transfer learning**).

La manera más sencilla de hacerlo sería fijar(freeze) los parámetros de las capas convolucionales, que contienen la mayoría de los parámetros. Y entrenar de nuevo las cabezas con datos que contengan las clases que queremos.

A grandes rasgos los pasos a seguir son:

1. Encontrar un modelo pre-entrenado que se ajuste aproximadamente a nuestras necesidades, teniendo en cuenta el parecido entre el modelo que realmente necesitamos(cuánto más parecido mejor), así como el tamaño etc...

2. Obtener datos, en el caso de no partir ya con imágenes etiquetadas, conseguir una pequeña muestra(eg: 100 instancias por clase) y solicitar/generar más en el futuro  en caso de ser necesario(asumiendo un coste apreciable en la obtención de los datos). Comprobar si existen fuentes de datos públicas de los datos que necesitamos, o de datos parecidos.(Amazon Turk)

3. Realizar el transfer learning comentado anteriormente y evaluar resultados.

4. En base a los resultados iterar modelo, hyperparametros, obtención de datos, preprocesamiento etc... hasta  cumplir los requerimientos.

Nota: Otra posible opción sería valorar usar modelos de zero-shot learning como clip de openai.

## Descripción de problemas que puedan surgir y medidas para reducir el riesgo

**Altos tiempos de inferencia y Alto coste computacional**

En el caso de necesitar utilizar el modelo en aplicaciones en tiempo real, el tiempo de inferencia puede ser demasiado grande, incluso usando **gpu**.

Valorar el uso de modelos más pequeños y rápidos(**yolov7**), mejor preprocesamiento.

Valorar también si es posible trabajar con imágenes más pequeñas o en escala de grises.

Además podemos realizar un diversas técnicas de **prunning** para reducir a posteriori el tamaño del modelo sin comprometer demasiado el rendimiento.

**Clases desbalanceadas**

El uso de 'focal loss' en la función de coste puede ser de ayuda.

Técnicas clásicas de oversampling.

Threshold Selection

**Bueno en entrenamiento malo en producción**

Es fundamental que la distribución de entrenamiento usada se parezca lo más posible al entorno de producción en el cuál se pretende implementar(condiciones de luz, punto de vista, textura de fondo, etc...). Esto podría dar lugar a data leakeage(eg: reconociendo el objeto por el fondo en vez de el sujeto) --> Adversarial Validation

**Mala calidad de los datos**

Exploración de los datos, resultados obtenidos del modelo, criterio experto del problema.

## 3. Estimación de datos necesarios, resultados y métricas esperadas

La cantidad de datos depende no sólo de la cantidad de imágenes sino del número de instancias del objecto a detectar dentro de esas imágenes, así como la variedad que exista en las imágenes. Cuánto mayor sea la variedad mejor, y más fácil será realizar data augmentation(rotaciones, ruido, translaciones ...).

Al utilizar transfer learning el coste computacional y la cantidad de datos necesarios para entrenar será mucho menor, aún así estimaría unas **~1000 instancias** por clase sería un buen comienzo.

Para evaluar el resultado una opción es usar el **iou(Jaccard Index)** para evaluar las bounding boxes y el **f1-score** para la clasificación.

Tener en cuenta el rendimiento humano esperable dados los datos.

**train vs test:**

En el caso de haber underfitting --> Modelos más grandes/sofisticados

En el caso de haber overfitting -->  Aplicar técnicas de regularización(en las cabezas), data augmentation, modelo más sencillo etc...

# Apartado 3 - Edge Computing

El uso de sistemas en el borde está asociado a mayor coste computacional, así que toma especial relevancia consideraciones en el tiempo de inferencia(y tamaño) del modelo. Probablemente sea muy interesante aplicar técnicas de prunning.

En el caso de realizar tmbn el entrenamiento sería necesario algun tipo de aprendizaje colaborativo(**federated learning**), que resolvería tmbn posibles problemas de privacidad de los datos.

Nota: Valorar solo para preprocesamiento al menos.
