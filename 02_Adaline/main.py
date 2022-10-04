import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import  make_classification
from sklearn.preprocessing import StandardScaler
from Adaline import Adaline

# Generació del conjunt de mostres
X, y = make_classification(n_samples=100, n_features=2, n_redundant=0, n_repeated=0,
                           n_classes=2, n_clusters_per_class=1, class_sep=1.5,
                           random_state=8)
y[y == 0] = -1  # La nostra implementació esta pensada per tenir les classes 1 i -1.


# TODO: Normalitzar les dades
scaler = StandardScaler()
xStandard = scaler.fit_transform(X)
# TODO: Entrenar usant l'algorisme de Batch gradient
adaline = Adaline(eta=0.001, n_iter=50)
adaline.fit(xStandard, y)
y_prediction = adaline.predict(xStandard)
# TODO: Mostrar els resultats
plt.figure(1)
# Dibuixam el núvol de punts (el parametre c indica que colorejam segons la classe)
plt.scatter(xStandard[:, 0], xStandard[:, 1], c=y)

# Dibuixem la recta. Usam l'equació punt-pendent
m = -adaline.w_[1] / adaline.w_[2]
origen = (0, -adaline.w_[0] / adaline.w_[2])
plt.axline(xy1=origen, slope=m)

### Extra: Dibuixam el nombre d'errors en cada iteracio de l'algorisme
plt.figure(2)
plt.plot(adaline.cost_, marker='o')
plt.xlabel('Epochs')
plt.ylabel('Number of misclassifications')
plt.show()