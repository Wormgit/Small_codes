import numpy as np
import matplotlib.pylab as plt

from sklearn.datasets import make_moons
from sklearn.preprocessing import StandardScaler
from sklego.mixture import GMMClassifier


#website
#https://scikit-lego.readthedocs.io/en/latest/mixture-methods.html


###Classification--------------
# n = 1000
# X, y = make_moons(n)
# X = X + np.random.normal(n, 0.12, (n, 2))
# X = StandardScaler().fit_transform(X)
# U = np.random.uniform(-2, 2, (10000, 2))
#
# n_components =16
#
# mod = GMMClassifier(n_components=n_components).fit(X, y)
#
# plt.figure(figsize=(8, 6))
# plt.subplot(211)
# plt.scatter(X[:, 0], X[:, 1], c=mod.predict(X), s=8)
# plt.title("classes of points")
#
# plt.subplot(212)
# plt.scatter(U[:, 0], U[:, 1], c=mod.predict_proba(U)[:, 1], s=8)
# plt.title("classifier boundary")
# plt.show()
# ###Classification--------------





###Outlier Detection-use GMMOutlierDetector score_samples--------
from sklego.mixture import GMMOutlierDetector

n = 1000
X = make_moons(n)[0] + np.random.normal(n, 0.12, (n, 2))
X = StandardScaler().fit_transform(X)

n_components = 10
mod = GMMOutlierDetector(n_components=n_components, threshold=0.95).fit(X)

plt.figure(figsize=(6, 6))

plt.subplot(211)
c=mod.score_samples(X) # in GMMOutlierDetector, the result is -log
plt.scatter(X[:, 0], X[:, 1], c=mod.score_samples(X), s=8)
plt.title(f"likelihood of points given mixture of {n_components} gaussian components")
plt.colorbar()

plt.subplot(212)
U = np.random.uniform(-2, 2, (10000, 2))
plt.scatter(U[:, 0], U[:, 1], c=mod.predict(U), s=8)
plt.title("outlier selection")
plt.colorbar()
plt.show()

#Instead of selection points that are beyond the likely quantile threshold one can also
#specify the number of standard deviations away from the most likely standard deviations a given point it.
#The outlier detection methods that we use are based on the likelihoods that come out of the estimated Gaussian Mixture.
#Depending on the setting you choose we have a different method for determining if a point is inside or outside the threshold.
plt.figure(figsize=(14, 3))
for i in range(1, 5):
    mod = GMMOutlierDetector(n_components=16, threshold=20, method="stddev").fit(X)
    plt.subplot(140 + i)
    plt.scatter(U[:, 0], U[:, 1], c=mod.predict(U), s=8)  #-1 (outlier) or +1 (normal).
    plt.title(f"outlier sigma={i}")
plt.show()





# plot, did not see it---------
# from scipy.stats import gaussian_kde
#
# score_samples = np.random.beta(220, 10, 3000)
# density = gaussian_kde(score_samples)
# likelihood_range = np.linspace(0.80, 1.0, 10000)
#
# index_max_y = np.argmax(density(likelihood_range))
# mean_likelihood = likelihood_range[index_max_y]
# new_likelihoods = score_samples[score_samples < mean_likelihood]
# new_likelihoods_std = np.sqrt(np.sum((new_likelihoods - mean_likelihood) ** 2)/(len(new_likelihoods) - 1))
#
# plt.figure(figsize=(8, 6))
# plt.subplot(211)
# plt.plot(likelihood_range, density(likelihood_range), 'k')
# xs = np.linspace(0.8, 1.0, 2000)
# plt.fill_between(xs, density(xs), alpha=0.8)
# plt.title("log-lik values from with GMM, quantile is based on blue part")
#
# plt.subplot(212)
# plt.plot(likelihood_range, density(likelihood_range), 'k')
# plt.plot([mean_likelihood, mean_likelihood], [0, density(mean_likelihood)], 'k--')
# xs = np.linspace(0.8, mean_likelihood, 2000)
# plt.fill_between(xs, density(xs), alpha=0.8)
# plt.title("log-lik values from with GMM, stddev is based on blue part")
# plt.show()
# plot, did not see it---------