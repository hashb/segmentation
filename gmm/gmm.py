#!/usr/bin/env python2
from __future__ import division, print_function
import numpy as np
import matplotlib.image as mpimg

def kmeans(k, img):
    x = img.copy()
    # random initialization
    means = np.random.random((k,1))

    convergence = float('inf')
    while convergence > 10**-8:
        dist = np.linalg.norm(means[:, np.newaxis] - x, axis=2)
        masks = [np.argmin(dist, axis=0) == _x for _x in range(k)]
        old_means = means.copy()
        means = np.array([np.mean(x[mask]) for mask in masks]).reshape((k,1))
        convergence = np.linalg.norm(np.abs(old_means) - np.abs(means))
    labels = np.zeros(img.shape)
    for i,mask in enumerate(masks):
        labels[mask] = i
    return means, labels

class GMM:
    def __init__(self, img, k, init='kmeans'):
        self.img = img.copy()
        self.img_mat = img.reshape((img.shape[0]*img.shape[1], 1))
        self.n = self.img_mat.shape[0]
        self.k = k

        if init == 'random':
            self.means = np.random.random((self.k,1))
        else:
            self.means = kmeans(self.k, self.img_mat)[0]

        self.sigma = np.ones((self.k,1))
        self.m_coeff = np.ones((self.k,1))/self.k
        self.gamma = np.zeros((self.n, self.k)) # responsibility matrix

        self.log_likelihood = []


    def fit(self, max_itr=2000):
        itr = 0
        convergence1 = False
        convergence2 = False
        img_mat_shuff = self.img_mat.copy()
        # np.random.shuffle(self.img_mat)

        while True:
            self.old_means = self.means.copy()
            self.old_sigma = self.sigma.copy()
            self.old_m_coeff = self.m_coeff.copy()

            # Expectation
            # gamma = self.expectation()
            for _i in range(self.k):
                self.gamma[:,_i] = self.m_coeff[_i] * self.prior(img_mat_shuff,self.means[_i], self.sigma[_i,:]).reshape(self.n)

            ll = np.sum(np.log(np.sum(self.gamma, axis = 1)))
            self.log_likelihood.append(ll)
            self.gamma = (self.gamma.T / np.sum(self.gamma, axis = 1)).T

            # Maximization
            N_k = np.sum(self.gamma, axis = 0)

            for _i in range(self.k):
                self.means[_i] = np.sum(self.gamma[:,_i] * img_mat_shuff.T, axis=1).T/N_k[_i]

                self.sigma[_i] = np.dot(np.multiply((img_mat_shuff - self.means[_i]).T, self.gamma[:,_i]), \
                                        img_mat_shuff - self.means[_i])/N_k[_i]

                self.m_coeff[_i] = N_k[_i]/self.n

            itr+=1
            print ("  Iteration %d out of %d\r" % (itr,max_itr), end='')

            convergence1 = np.allclose(self.means, self.old_means) and \
                          np.allclose(self.sigma, self.old_sigma) and \
                          np.allclose(self.m_coeff, self.old_m_coeff)
            if convergence1: break
            if len(self.log_likelihood) >2:
                convergence2 = self.log_likelihood[-1] - self.log_likelihood[-3] < 10**-5
            if convergence2: break
            if itr > max_itr: break
        print ("Fitting complete in {} Iteration".format(itr))


    def prior(self, x, mu, sigma):
        a1 = (2*sigma*np.pi)**-0.5
        a2 = np.exp(-0.5 * sigma**-1 * (x-mu)**2)
        return np.array(a1*a2)

    def predict(self, x):
        probs = []
        for mean, sigma, m_coeff in zip(self.means, self.sigma, self.m_coeff):
            prob = m_coeff * self.prior(x, mean, sigma).reshape(self.n)
            probs.append(prob)
        probs = np.array(probs)
        label = np.argmax(probs, axis=0)
        return label

    def segment(self):
        # y = np.array([self.predict(_i)[0] for _i in self.img_mat.tolist()])
        label = self.predict(self.img_mat.copy())
        return label.reshape(self.img.shape)

    def run(self, num_itr):
        # returns a segmented image
        self.fit(num_itr)
        print ("Mean: ", self.means)
        print ("")
        print ("Variance: ", self.sigma)
        print ("")
        print ("Weights", self.m_coeff)

        return self.segment()

def rgb2gray(img):
    # source: https://stackoverflow.com/a/12201744
    # weights = [0.299, 0.587, 0.114]  # grayscale
    weights = [0.21, 0.72, 0.07]  # luminosity
    return np.dot(img[...,:3], weights)/255.

if __name__ == '__main__':
    x = rgb2gray(mpimg.imread('1.jpg'))
    y = x.reshape((x.shape[0]*x.shape[1], 1))
    g = GMM(y, 2)
    g.run(1000)





