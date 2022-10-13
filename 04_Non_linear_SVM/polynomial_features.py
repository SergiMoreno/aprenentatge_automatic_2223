import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from sklearn.preprocessing import PolynomialFeatures
from sklearn.svm import SVC

def kernel_poly(x1, x2, degree=3):
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
svm = SVC(C=1.0, kernel='poly', gamma=1) # per defecte té degree=3
# Train the classifier
svm.fit(X_transformed, y_train)
y_predicted = svm.predict(X_test_transformed)

#
polysvm = SVC(C=1.0, kernel=kernel_poly)
polysvm.fit(X_transformed, y_train)
polyy_predicted = polysvm.predict(X_test_transformed)

#
poly = PolynomialFeatures(degree=3)
X_poly = poly.fit_transform(X_transformed)
X_test_poly = poly.transform(X_test_transformed)
svm_linear = SVC(C=1.0, kernel='linear', gamma=1)
svm_linear.fit(X_poly, y_train)
py_predicted = svm_linear.predict(X_test_poly)

differences = (y_predicted - y_test)
errors = np.count_nonzero(differences)
metric = (len(y_predicted)-errors)/len(y_predicted)

print(f'Rati d\'acerts en el bloc de predicció (POLY): {metric}')

polydifferences = (polyy_predicted - y_test)
polyerrors = np.count_nonzero(polydifferences)
polymetric = (len(polyy_predicted)-polyerrors)/len(polyy_predicted)

print(f'Rati d\'acerts en el bloc de predicció (KERNEL_POLY): {polymetric}')

result_diff = metric - polymetric
print(f'Diferència de ratis: {result_diff}')

pdifferences = (py_predicted - y_test)
perrors = np.count_nonzero(pdifferences)
polymetric = (len(py_predicted)-perrors)/len(py_predicted)

print(f'Rati d\'acerts en el bloc de predicció (POLYNOMIAL): {polymetric}')

result_diff_poly = metric - polymetric
print(f'Diferència de rati amb PolynomialFeatures: {result_diff_poly}')