import matplotlib

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

from sklearn import neighbors, datasets


n_neighbors = 5

# ornek dataset olustur,
iris = datasets.load_iris()  #iris  type: Bunch

# grafik icin datalari hazirla
X = iris.data[:, :2]
y = iris.target
h = .02

# grafik icin renklendirmeleri ayarla
cmap_light = ListedColormap(['#F5A9D0', '#AAFFAA', '#00AAFF'])
cmap_bold = ListedColormap(['#FF0000', '#00FF00', '#00AAFF'])

# KNN icin bir instance(ornekleme) olusturacagiz
# weights default degeri uniform
clf = neighbors.KNeighborsClassifier(n_neighbors, weights='distance')
clf.fit(X, y)

# min, max ve limiti hesaplayalim
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))

# hazirlanan datadan cikarim yaptiralim
Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])

# cizilecek grafigin renk ve hesaplanmis datalari vereleim
Z = Z.reshape(xx.shape)
plt.figure()
plt.pcolormesh(xx, yy, Z, cmap=cmap_light)

# matplotlib kutuphanesi sayesinde x, y kordinatlarini, renkleri vererek cizim yaptiralim
plt.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap_bold)
plt.xlim(xx.min(), xx.max())
plt.ylim(yy.min(), yy.max())

# Cizimi goster
plt.show()
