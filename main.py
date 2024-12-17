from sklearn.datasets import load_iris # Carga el conjunto de datos Iris
from sklearn.model_selection import train_test_split # Divide los datos en conjuntos de prueba y entrenamiento para evaluar el modelo
from sklearn.neighbors import KNeighborsClassifier # Carga el clasificador de vecinos mas cercanos (KNN)
from sklearn.metrics import accuracy_score # Calcula la exactitud del clasificador y del modelo

# Cargar el conjunto de datos Iris
iris = load_iris() # Carga el conjunto de datos Iris
X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.2, random_state=42) # Dividir los datos en conjuntos de entrenamiento y prueba
# test_size es el porcentaje de los datos que se usan para prueba (20%) y el random_state es un semilla para la aleatoriedad y asegura que la división sea la misma cada vez que se ejecuta el código

# Crear el clasificador de vecinos más cercanos
clf = KNeighborsClassifier(n_neighbors=3) # Crea un clasificador de vecinos más cercanos con 3 vecinos para hacer predicciones

# Entrenar el clasificador
clf.fit(X_train, y_train) # Entrena el clasificador con los datos de entrenamiento, aprende a clasificar las flores

# Predecir las etiquetas para los datos de prueba
y_pred = clf.predict(X_test) # Predecir las etiquetas para los datos de prueba, en este caso, las flores y su tipo

# Calcular la exactitud del clasificador
accuracy = accuracy_score(y_test, y_pred) # Compara las predicciones con los datos reales y calcula la exactitud, en este caso, la exactitud de la clasificación
print("Exactitud:", accuracy)
