from sklearn.model_selection import cross_val_score
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression

# Cargar el conjunto de datos Iris
iris = load_iris()

# Crear el clasificador de regresión logística
clf = LogisticRegression()

# Realizar validación cruzada
scores = cross_val_score(clf, iris.data, iris.target, cv=5)

print("Precisión de validación cruzada:", scores)