import matplotlib.pyplot as plt
import numpy as np

x = [1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000]

plt.plot(x, [84.5, 88.6, 90.2, 90.9, 92.1, 91.9, 92.8, 92.0, 93.1]) # knn, k=1
plt.plot(x, [83.9, 90.0, 90.8, 92.0, 92.4, 92.8, 94.0, 93.8, 93.6]) # knn, k=5
plt.plot(x, [80.1, 85.0, 88.0, 88.6, 89.0, 90.1, 90.3, 91.5, 91.0]) # knn, k=30
plt.plot(x, [88.1, 91.8, 94.0, 94.1, 94.3, 94.9, 94.5, 94.3, 95.0]) # max-likelihood
plt.plot(x, [94.5, 94.6, 94.5, 94.5, 94.6, 94.7, 94.6, 94.8, 95.2]) # SVM

plt.legend(['KNN, k=1', 'KNN, k=5', 'KNN, k=30', 'Max-Likelihood', 'SVM'], loc='lower right')

plt.title('Model Results for Handwritten Digit Classification', fontsize=18)

plt.xlabel('Training Samples', fontsize=14)
plt.ylabel('Accuracy', fontsize=14)


plt.show()