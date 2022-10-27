import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
import numpy as np
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture

MIN_ACCUMULATIVE_DIFF = 0.001
MIN_VARIANCE_PERCENTAGE = 0.98

def plot_digits(data):
    fig, ax = plt.subplots(10, 10, figsize=(10, 10),
                           subplot_kw=dict(xticks=[], yticks=[]))
    fig.subplots_adjust(hspace=0.05, wspace=0.05)
    for i, axi in enumerate(ax.flat):
        im = axi.imshow(data[i].reshape(8, 8), cmap='binary')
        im.set_clim(0, 16)

def flatten_curve(variance):
    for i in range(1, len(variance)):
        diff = variance[i] - variance[i-1]
        if diff < MIN_ACCUMULATIVE_DIFF and variance[i] > MIN_VARIANCE_PERCENTAGE:
            return (i+1)
    return len(variance)


digits = datasets.load_digits()
plot_digits(digits.data[:100, :])

plt.show()
#print(digits.data.shape)

#scaler = StandardScaler()
#Xdata = digits.data
#X = scaler.fit_transform(Xdata)
X = digits.data

pca = PCA(random_state=0)
digits_pca = pca.fit_transform(X)
#digits_pca_variance = pca.explained_variance_ amb es ratio queda més estètic
digits_pca_variance = pca.explained_variance_ratio_

# plt.plot(digits_pca_variance) show of the variance from high to low
digits_pca_accumulative = np.cumsum(digits_pca_variance)

new_components = flatten_curve(digits_pca_accumulative)
print(new_components)
                                                   # 0.005 i 0.98 -> 36
# plt.plot(digits_pca_accumulative[:new_components]) # 0.001 i 0.98 -> 46
# plt.axvline(x = new_components, ymax=100, ymin=-100) retxa vertical
# plt.show()

plt.plot(digits_pca_accumulative) # acumulative show of the variance
plt.axvline(x = new_components, ymax=100, ymin=-100, color='r')
plt.axhline(y=digits_pca_accumulative[new_components], color='y')
plt.show()

pca_reduced = PCA(n_components=new_components, random_state=0)
digits_pca_reduced = pca_reduced.fit_transform(X)
digits_pca_reduced_variance = pca_reduced.explained_variance_ratio_
digits_pca_reduced_accumulative = np.cumsum(digits_pca_reduced_variance)
plt.plot(digits_pca_reduced_accumulative)
plt.show()

# plot de gaussiana entre 1 i 10
arrayGaussian = []
for n in range(1, 11):
    gaussian = GaussianMixture(n_components=n, random_state=0)
    digits_gaussian = gaussian.fit(digits_pca_reduced)
    digits_gaussian_bic = gaussian.bic(digits_pca_reduced)
    #print("BIC per", n, " components :", digits_gaussian_bic)
    arrayGaussian.append(digits_gaussian_bic)
plt.plot(arrayGaussian)
plt.show()
# min = gaussian7

gaussian = GaussianMixture(n_components=7, random_state=0).fit(digits_pca_reduced)
hundred_samples = gaussian.sample(n_samples=100)
inversed_samples = pca_reduced.inverse_transform(hundred_samples[0])
plot_digits(inversed_samples[:100, :])
plt.show()
#print(inversed_samples.shape)