import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
import numpy as np
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn.datasets import fetch_lfw_people

# MIN_ACCUMULATIVE_DIFF = 0.001
# MIN_VARIANCE_PERCENTAGE = 0.98

def plot_digits(data):
    fig, ax = plt.subplots(10, 10, figsize=(10, 10),
                           subplot_kw=dict(xticks=[], yticks=[]))
    fig.subplots_adjust(hspace=0.05, wspace=0.05)
    for i, axi in enumerate(ax.flat):
        im = axi.imshow(data[i].reshape(8, 8), cmap='binary')
        im.set_clim(0, 16)

def plot_gallery(images, h, w, n_row=3, n_col=4):
    """Helper function to plot a gallery of portraits"""
    plt.figure(figsize=(1.8 * n_col, 2.4 * n_row))
    plt.subplots_adjust(bottom=0, left=0.01, right=0.99, top=0.90, hspace=0.35)
    for i in range(n_row * n_col):
        plt.subplot(n_row, n_col, i + 1)
        plt.imshow(images[i].reshape((h, w)), cmap=plt.cm.gray)
        # plt.title(titles[i], size=12)
        plt.xticks(())
        plt.yticks(())

def flatten_curve(variance, min_accumulative_diff, min_variance_percentage):
    for i in range(1, len(variance)):
        diff = variance[i] - variance[i-1]
        # if diff < MIN_ACCUMULATIVE_DIFF and variance[i] > MIN_VARIANCE_PERCENTAGE:
        if diff < min_accumulative_diff and variance[i] > min_variance_percentage:
            return i+1
    return len(variance)

# SHOW ORIGINAL DIGITS
digits = datasets.load_digits()
plot_digits(digits.data[:100, :])
plt.show()
# print(digits.data.shape)

# PCA WITH ALL THE COMPONENTS

# scaler = StandardScaler()
# Xdata = digits.data
# X = scaler.fit_transform(Xdata)
X = digits.data

pca = PCA(random_state=0)
digits_pca = pca.fit_transform(X)
# digits_pca_variance = pca.explained_variance_
digits_pca_variance = pca.explained_variance_ratio_ # amb es ratio queda més estètic

# plt.plot(digits_pca_variance) show of the variance from high to low
digits_pca_accumulative = np.cumsum(digits_pca_variance)
new_components = flatten_curve(digits_pca_accumulative, min_accumulative_diff=0.001, min_variance_percentage=0.98)
print("Digits - new number of components:", new_components)

# SHOW ACCUMULATIVE VARIANCE WITH THE POINT MARKED WHERE WE STOP
plt.plot(digits_pca_accumulative) # acumulative show of the variance
plt.axvline(x=new_components, ymax=100, ymin=-100, color='r')
plt.axhline(y=digits_pca_accumulative[new_components], color='y')
plt.title("Digits - Accumulative variance with all the components")
plt.show()

# PCA ONLY WITH A SUBSET OF COMPONENTS

pca_reduced = PCA(n_components=new_components, random_state=0)
digits_pca_reduced = pca_reduced.fit_transform(X)
digits_pca_reduced_variance = pca_reduced.explained_variance_ratio_
digits_pca_reduced_accumulative = np.cumsum(digits_pca_reduced_variance)
plt.plot(digits_pca_reduced_accumulative)
plt.title("Digits - Accumulative variance with " + str(new_components) + " components")
plt.show()

# plot de gaussiana entre 1 i 10
arrayGaussian = []
for n in range(1, 11):
    gaussian = GaussianMixture(n_components=n, random_state=0)
    digits_gaussian = gaussian.fit(digits_pca_reduced)
    digits_gaussian_bic = gaussian.bic(digits_pca_reduced)
    arrayGaussian.append(digits_gaussian_bic)
plt.plot(arrayGaussian)
plt.axhline(y=min(arrayGaussian), color='r')
plt.title("Digits - Gaussian from 1 to 10")
plt.show()

# SELECT THE MINIMUM GAUSSIAN TO PRODUCE SAMPLES TO TRANSFORM TO THE ORIGINAL DIMENSIONALITY
# adding +1 as the xLab is numbered from 0 to 9, not from 1 to 10
gaussian = GaussianMixture(n_components=8, random_state=0).fit(digits_pca_reduced)
hundred_samples = gaussian.sample(n_samples=100)
inversed_samples = pca_reduced.inverse_transform(hundred_samples[0])
plot_digits(inversed_samples[:100, :])
plt.show()

# WE DO THE SAME, NOW WITH THE FACES DATASET
# SCALE THE DATA
lfw_people = fetch_lfw_people()
scaler = StandardScaler()
xF = lfw_people.data
xFaces = scaler.fit_transform(xF)
# xFaces = lfw_people.data

# SHOW ORIGINAL FACES
n_samples, h, w = lfw_people.images.shape
plot_gallery(lfw_people.images, h, w)
plt.show()
# print(xFaces.shape)

# PCA WITH ALL THE COMPONENTS
faces_pca = pca.fit_transform(xFaces)
faces_pca_variance = pca.explained_variance_ratio_
faces_pca_accumulative = np.cumsum(faces_pca_variance)

faces_new_components = flatten_curve(faces_pca_accumulative, min_accumulative_diff=0.0001, min_variance_percentage=0.98)
print("Faces - new number of components:", faces_new_components)

# SHOW ACCUMULATIVE VARIANCE WITH THE POINT MARKED WHERE WE STOP
plt.plot(faces_pca_accumulative)
plt.axvline(x=faces_new_components, ymax=100, ymin=-100, color='r')
plt.axhline(y=faces_pca_accumulative[faces_new_components], color='y')
plt.title("Faces - Accumulative variance with all the components")
plt.show()

# PCA ONLY WITH A SUBSET OF COMPONENTS

pca_reduced = PCA(n_components=faces_new_components, random_state=0)
faces_pca_reduced = pca_reduced.fit_transform(xFaces)
faces_pca_reduced_variance = pca_reduced.explained_variance_ratio_
faces_pca_reduced_accumulative = np.cumsum(faces_pca_reduced_variance)
plt.plot(faces_pca_reduced_accumulative)
plt.title("Faces - Accumulative variance with " + str(faces_new_components) + " components")
plt.show()

# plot de gaussiana entre 1 i 10
# ERROR WITH KMEANS MEMORY LEAK
arrayGaussian = []
for n in range(1, 11):
    gaussian = GaussianMixture(n_components=n, random_state=0)
    faces_gaussian = gaussian.fit(faces_pca_reduced)
    faces_gaussian_bic = gaussian.bic(faces_pca_reduced)
    arrayGaussian.append(faces_gaussian_bic)
plt.plot(arrayGaussian)
plt.axhline(y=min(arrayGaussian), color='r')
plt.title("Faces - Gaussian from 1 to 10")
plt.show()

# SELECT THE MINIMUM GAUSSIAN TO PRODUCE SAMPLES TO TRANSFORM TO THE ORIGINAL DIMENSIONALITY
gaussian = GaussianMixture(n_components=1, random_state=0).fit(faces_pca_reduced)
hundred_samples = gaussian.sample(n_samples=100)
inversed_samples = pca_reduced.inverse_transform(hundred_samples[0])
plot_gallery(inversed_samples, h, w)
plt.show()
# FORA STANDARD
# 0.0000001 i 0.99-> 1704 components, minGaussian=2
# 0.0001 i 0.98->317 components, minGaussian=10