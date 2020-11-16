import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from matplotlib.colors import ListedColormap
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier


# path_to_data = 'https://raw.githubusercontent.com/jeshan49/EEMP2019/master/content/part-5/part-5-1/Default.csv'
# df = pd.read_csv(path_to_data, index_col=0)
# df['default'] = pd.get_dummies(df['default'], drop_first=True)
# X, test_X, y, test_y = train_test_split(df[['balance', 'income']], df['default'], test_size=0.9, random_state=181)
# qda = QuadraticDiscriminantAnalysis().fit(test_X,test_y)
# h = 10  # step size in the mesh
# x_min, x_max = test_X['balance'].min() - 100, test_X['balance'].max() + 1
# y_min, y_max = test_X['income'].min() - 1, test_X['income'].max() + 1
# xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
# YY = qda.predict(np.c_[xx.ravel(), yy.ravel()])
# YY = YY.reshape(xx.shape)
# fig, axs = plt.subplots(1, 2, figsize=(15, 6), tight_layout=True)
# axs[0].scatter(df['balance'][df['default'] == 1], df['income'][df['default'] == 1], color='red', label='Yes')
# axs[0].scatter(df['balance'][df['default'] == 0], df['income'][df['default'] == 0], color='blue', alpha=0.5, label='No')


# axs[0].contour(xx, yy, YY, [0.5], colors='black', linestyles='dashed')
# axs[0].set_title('Population - Default')
# axs[1].scatter(X['balance'][y == 1],X['income'][y == 1], color='red', label='Yes')
# axs[1].scatter(X['balance'][y == 0],X['income'][y == 0], color='blue', alpha=0.5, label='No')
# axs[1].contour(xx, yy, YY, [0.5], colors='black', linestyles='dashed')
# axs[1].set_title('Sample - Default')
# for ax in axs:
#     ax.legend()
#     ax.set_ylabel('Income')
#     ax.set_xlabel('Balance')
#     ax.set_xlim(-100, 3000)
#     ax.set_ylim(-1000, 75000)
# plt.show()


# # Code that generates plots with Linear regression and KNN decision boundaries along with the optimal decision boundary
# lda = LinearDiscriminantAnalysis()
# knn = KNeighborsClassifier(n_neighbors=1, n_jobs=-1)
# lda.fit(X, y).predict(X)
# knn.fit(X, y).predict(X)
# cmap_light = ListedColormap(['#AAAAFF', '#FFAAAA'])
# fig, axs = plt.subplots(1, 2, figsize=(15, 6), tight_layout=True)
# clfs = [lda, knn]
# clfs_names = ['Linear Regression', '1NN Classifier']
# for i in range(len(clfs)):
#     Z = clfs[i].predict(np.c_[xx.ravel(), yy.ravel()])
#     Z = Z.reshape(xx.shape)
#     axs[i].pcolormesh(xx, yy, Z, cmap=(cmap_light))
#     YY = qda.predict(np.c_[xx.ravel(), yy.ravel()])
#     YY = YY.reshape(xx.shape)
#     axs[i].set_title(clfs_names[i])
#     axs[i].contour(xx, yy, YY, [0.5], colors='black', linestyles='dashed')
#     axs[i].scatter(X['balance'][y == 1],X['income'][y == 1], color='red', label='Yes')
#     axs[i].scatter(X['balance'][y == 0],X['income'][y == 0], color='blue', alpha=0.2, label='No')
#     axs[i].set_xlabel('Balance')
#     axs[i].set_xlabel('Balance')
#     plt.show();
