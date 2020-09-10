from sklearn import datasets
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.metrics import accuracy_score
import diffprivlib.models as models
import matplotlib.pyplot as plt


dataset = datasets.load_iris()
X_train, X_test, y_train, y_test = train_test_split(dataset.data, dataset.target, test_size=0.2)

from sklearn.naive_bayes import GaussianNB as GNB
#clf = GaussianNB()
clf = models.GaussianNB(0.1)
clf.fit(X_train, y_train)

clf.predict(X_test)



print("Test accuracy: %f" % accuracy_score(y_test, clf.predict(X_test)))



epsilons = np.logspace(-2,2, 50)
bounds = [(4.3, 7.9), (2.0, 4.4), (1.1, 6.9), (0.1, 2.5)]
accuracy = list()
accuracy1=list()

for epsilon in epsilons:
    clf = models.GaussianNB(bounds=bounds, epsilon=epsilon)
    clf.fit(X_train, y_train)
    clf1 = GNB()
    clf1.fit(X_train, y_train)
    accuracy.append(accuracy_score(y_test, clf.predict(X_test)))
    accuracy1.append(accuracy_score(y_test, clf1.predict(X_test)))
print(epsilons, accuracy)
print(epsilons, accuracy1)
plt.semilogx(epsilons, accuracy)
plt.semilogx(epsilons, accuracy1)
plt.title("Differentially private Naive Bayes accuracy")
plt.xlabel("epsilon")
plt.ylabel("Accuracy")
plt.show()
print('finished!!')
exit()