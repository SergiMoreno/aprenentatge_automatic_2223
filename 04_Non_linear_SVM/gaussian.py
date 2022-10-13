import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from scipy.spatial import distance_matrix

from sklearn.svm import SVC

def kernel_gauss(x1, x2, gamma=1):
    return np.exp(-gamma * (distance_matrix(x1, x2) ** 2))
    #return np.exp(np.dot(-gamma,(distance_matrix(x1, x2)**2)))

# Generació del conjunt de mostres
X, y = make_classification(n_samples=400, n_features=5, n_redundant=0, n_repeated=0,
                           n_classes=2, n_clusters_per_class=1, class_sep=0.5,
                           random_state=9)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

# Els dos algorismes es beneficien d'estandaritzar les dades
scaler = StandardScaler()
X_transformed = scaler.fit_transform(X_train)
X_test_transformed = scaler.transform(X_test)

# Create a SVC classifier using an RBF kernel
svm = SVC(kernel='rbf', random_state=0, gamma=1, C=1)
# Train the classifier
svm.fit(X_transformed, y_train)
y_predicted = svm.predict(X_test_transformed)

gausssvm = SVC(kernel=kernel_gauss, random_state=0, C=1)
gausssvm.fit(X_transformed, y_train)
gaussy_predicted = gausssvm.predict(X_test_transformed)

differences = (y_predicted - y_test)
errors = np.count_nonzero(differences)
metric = (len(y_predicted)-errors)/len(y_predicted)

print(f'Rati d\'acerts en el bloc de predicció: {metric}')

gaussdifferences = (gaussy_predicted - y_test)
gausserrors = np.count_nonzero(gaussdifferences)
gaussmetric = (len(gaussy_predicted)-gausserrors)/len(gaussy_predicted)

print(f'Rati d\'acerts en el GAUSSIAN bloc de predicció: {gaussmetric}')

result_diff = metric - gaussmetric
print(f'Diferència de ratis: {result_diff}')