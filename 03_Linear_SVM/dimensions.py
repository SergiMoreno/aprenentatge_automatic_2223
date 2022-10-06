import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from Adaline import Adaline
from sklearn.svm import SVC


# Generació del conjunt de mostres
X, y = make_classification(n_samples=400, n_features=5, n_redundant=0, n_repeated=0,
                           n_classes=2, n_clusters_per_class=1, class_sep=1,
                           random_state=9)

# Separar les dades: train_test_split

# TODO
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=104, test_size=0.25, shuffle=True)

# Estandaritzar les dades: StandardScaler

# TODO
scaler = StandardScaler()
X_transformed = scaler.fit_transform(X_train)

# Entrenam una SVM linear (classe SVC)

# TODO
linearSVC = SVC(C=1000, kernel='linear')
linearSVC.fit(X_transformed, y_train, None)

# Prediccio
# TODO
y_prediction = linearSVC.predict(X_test)

# Metrica
# TODO
result = (y_test - y_prediction)
errors = len(np.nonzero(result))
metrica = (len(y_test) - errors)/(len(y_test))
print("Mètrica: %s" % metrica)