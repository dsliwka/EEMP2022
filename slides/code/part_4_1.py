import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
# plt.rc('axes', titlesize=45)
# plt.rc('axes', labelsize=40)
# plt.rc('legend', fontsize=40)
# plt.rc('xtick', labelsize=30)
# plt.rc('ytick', labelsize=30)
# import matplotlib as mpl
# mpl.rcParams['lines.linewidth'] = 2


path_to_data = 'https://raw.githubusercontent.com/jeshan49/EEMP2019/master/content/part-5/part-5-1/income.csv'
df = pd.read_csv(path_to_data)


# Plot of population of income as a function of age
plt.clf()
plt.scatter(x=df['age'],y=df['income'])
plt.title('Population of income based on age')
plt.xlabel('age')
plt.ylabel('income')
plt.savefig('../figures/fig4_1.png')

# Plot of population of income as a function of age
plt.clf()
sns.regplot(x='age', y='income', data=df, order=2, ci=None,
            scatter_kws={'color':'blue'},
            line_kws={'color':'red', 'ls':'--'}).set_title('Population of income based on age')
plt.legend(labels=['f(x)=E[y|x]'])
plt.savefig('../figures/fig4_2.png')

# Plot of sample of income as a function of age
plt.clf()
X, _, y, _ = train_test_split(df['age'], df['income'], test_size=0.99, random_state=181)
plt.figure(figsize=(20,10))
sns.regplot(x='age', y='income', data=df, order=2, ci=None, scatter=None, line_kws={'color':'red', 'ls':'--'}).set_title('income as a function of age')
plt.scatter(X, y)
plt.legend(labels=['f(x)=E[y|x]'])
plt.savefig('../figures/fig4_3.png')

from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
reg = LinearRegression()
knn = KNeighborsRegressor(n_neighbors=1)
X, _, y, _ = train_test_split(df['age'], df['income'], test_size=0.99)
range_X = np.linspace(df['age'].min(), df['age'].max(), 1000)
X, y, range_X = X[:, np.newaxis], y[:, np.newaxis], \
                np.linspace(df['age'].min(), df['age'].max(), 1000)[:, np.newaxis]
y_hat_reg, y_hat_knn = reg.fit(X,y).predict(range_X), knn.fit(X,y).predict(range_X)
fig, axs = plt.subplots(1, 2, figsize=(15, 6), tight_layout=True)
title, fit = ['Linear Regression', '1NN Regression'], ['Underfitting', 'Overfitting']
axs[0].plot(range_X, y_hat_reg, c='g')
axs[1].step(range_X, y_hat_knn, c='g')
for i in range(len(axs)):
    axs[i].scatter(X,y)
    axs[i].set_title(title[i])
    axs[i].set_xlabel('X')
    axs[i].set_ylabel('y')
    sns.regplot(x='age', y='income', data=df, order=2, ci=None, scatter=None, ax=axs[i], line_kws={'color':'red', 'ls':'--'})
    axs[i].text(40, 20000, fit[i], fontsize=18)
    axs[i].legend(labels=['$\hat{f}(x)$', '$f(x)$'])
    axs[i].set_xlim(df['age'].min(), df['age'].max())
    axs[i].set_ylim(df['income'].min(), df['income'].max())
plt.savefig('../figures/fig4_4.png')
