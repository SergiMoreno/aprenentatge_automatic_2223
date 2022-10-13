import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from sklearn.svm import SVC

def kernel_poly(x1, x2, degree=8):
    return (x1.dot(x2.T)) ** degree

# Generació del conjunt de mostres
X, y = make_classification(n_samples=400, n_features=5, n_redundant=0, n_repeated=0,
                           n_classes=2, n_clusters_per_class=1, class_sep=0.5,
                           random_state=9)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

# Els dos algorismes es beneficien d'estandaritzar les dades
scaler = StandardScaler()
X_transformed = scaler.fit_transform(X_train)
X_test_transformed = scaler.transform(X_test)

# Create a SVC classifier using an Polynomic kernel
svm = SVC(C=1.0, kernel='poly', gamma=1, degree=8) # per defecte té degree=3
# Train the classifier
svm.fit(X_transformed, y_train)
y_predicted = svm.predict(X_test_transformed)

polysvm = SVC(C=1.0, kernel=kernel_poly)
polysvm.fit(X_transformed, y_train)
polyy_predicted = polysvm.predict(X_test_transformed)

differences = (y_predicted - y_test)
errors = np.count_nonzero(differences)
metric = (len(y_predicted)-errors)/len(y_predicted)

print(f'Rati d\'acerts en el bloc de predicció de LLIBRERÍA: {metric}')

polydifferences = (polyy_predicted - y_test)
polyerrors = np.count_nonzero(polydifferences)
polymetric = (len(polyy_predicted)-polyerrors)/len(polyy_predicted)

print(f'Rati d\'acerts en el POLYNOMIC bloc de predicció: {polymetric}')

result_diff = metric - polymetric
print(f'Diferència de ratis: {result_diff}')