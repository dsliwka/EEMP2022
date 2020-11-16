import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from ipywidgets import IntSlider
from ipywidgets.embed import embed_minimal_html
slider = IntSlider(value=40)
embed_minimal_html('export.html', views=[slider], title='Widgets export')

path_to_data = 'https://raw.githubusercontent.com/jeshan49/EEMP2019/master/content/part-5/part-5-1/income.csv'
df = pd.read_csv(path_to_data)

# Plot of population of income as a function of age
sns.regplot(x='age', y='income', data=df, order=2, ci=None,
            scatter_kws={'color':'blue', 'alpha': 0.01},
            line_kws={'color':'red', 'ls':'--'}).set_title('income as a function of age')
plt.legend(labels=['f(x)=E[y|x]'])
plt.savefig('../figures/population_income_age.png')

# Plot of sample of income as a function of age
X, _, y, _ = train_test_split(df['age'], df['income'], test_size=0.99, random_state=181)
plt.figure(figsize=(20,10))
plt.scatter(X, y)
sns.regplot(x='age', y='income', data=df, order=2, ci=None, scatter=None, line_kws={'color':'red', 'ls':'--'}).set_title('income as a function of age')
plt.show();


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
plt.show();


from sklearn.metrics import mean_squared_error as mse
from ipywidgets import interactive

X, _, y, _ = train_test_split(df['age'], df['income'], test_size=0.99, random_state=181)
range_X = np.linspace(df['age'].min(), df['age'].max(), 1000)
X, y, range_X = X[:, np.newaxis], y[:, np.newaxis], \
                np.linspace(df['age'].min(), df['age'].max(), 1000)[:, np.newaxis]
mse_f_train = mse(y, 2000*X - 20*X**2)
mse_f_test = mse(df['income'], 2000*df['age'] - 20*df['age']**2)
rrange = np.arange(1, 101, 3)

mses_train = np.empty(34)
mses_train[:] = np.nan

mses_test = np.empty(34)
mses_test[:] = np.nan

mses_test_f = np.empty(34)
mses_test_f[:] = np.nan

def plot_with_slider(k):
    knn = KNeighborsRegressor(n_neighbors=k)
    y_hat_knn = knn.fit(X, y).predict(range_X)
    fig, axs = plt.subplots(1, 2, figsize=(15, 6), tight_layout=True)
    axs[0].scatter(X, y, c='b', label='Sample')
    axs[0].scatter(df['age'], df['income'], c='grey', alpha=0.1, label='Population')
    mse_train = mse(y,knn.fit(X, y).predict(X))
    mse_test = mse(df['income'][:, np.newaxis],knn.fit(X, y).predict(df['age'][:, np.newaxis]))
    mses_train[k//3] = mse_train
    mses_test[k//3] = mse_test
    mses_test_f[k//3] = mse_f_test
    sns.regplot(x='age', y='income', data=df, order=2, ci=None, scatter=None, ax=axs[0], line_kws={'color':'red', 'ls':'--'})
    axs[0].step(range_X, y_hat_knn, c='g', label='$\hat{f}$')
    axs[0].set_title('{}NN Regression'.format(k))
    axs[0].legend(loc='lower right')
    axs[1].plot(rrange,mses_train, label= '{0}$NN$ Training MSE: {1:,.2f}'.format(k,mse_train))
    axs[1].plot(rrange,mses_test, label='${0}NN$ Test MSE: {1:,.2f}'.format(k,mse_test))
    axs[1].plot(rrange,mses_test_f, c='black', ls='--', label='$f$ Test MSE: {0:,.2f}'.format(mse_f_test))
    axs[1].set_xticks(rrange)
    axs[1].set_ylim(0, 80000000)
    axs[1].legend(loc='upper center')
    axs[1].set_xlabel('# of neighbors')
    axs[1].set_ylabel('MSE')
    fig.canvas.draw_idle()

interactive_plot = interactive(plot_with_slider, k=(1, 100, 3))
interactive_plot
embed_minimal_html('aexport.html', views=[interactive_plot], title='Widgets export')




import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

# Create some random data
x = np.linspace(0,100,1000)
y = np.sin(x) * np.cos(x)

left, bottom, width, height = 0.15, 0.02, 0.7, 0.10

fig, ax = plt.subplots()

plt.subplots_adjust(left=left, bottom=0.25) # Make space for the slider

ax.plot(x,y)

# Set the starting x limits
xlims = [0, 1]
ax.set_xlim(*xlims)

# Create a plt.axes object to hold the slider
slider_ax = plt.axes([left, bottom, width, height])
# Add a slider to the plt.axes object
slider = Slider(slider_ax, 'x-limits', valmin=0.0, valmax=100.0, valinit=xlims[1])

# Define a function to run whenever the slider changes its value.
def update(val):
    xlims[1] = val
    ax.set_xlim(*xlims)

    fig.canvas.draw_idle()

# Register the function update to run when the slider changes value
slider.on_changed(update)

plt.show()



from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
path_to_data = 'https://raw.githubusercontent.com/jeshan49/EEMP2019/master/content/part-5/part-5-1/Default.csv'
df = pd.read_csv(path_to_data, index_col=0)
df['default'] = pd.get_dummies(df['default'], drop_first=True)
X, test_X, y, test_y = train_test_split(df[['balance', 'income']], df['default'], test_size=0.9, random_state=181)
qda = QuadraticDiscriminantAnalysis().fit(test_X,test_y)
h = 10  # step size in the mesh
x_min, x_max = test_X['balance'].min() - 100, test_X['balance'].max() + 1
y_min, y_max = test_X['income'].min() - 1, test_X['income'].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
YY = qda.predict(np.c_[xx.ravel(), yy.ravel()])
YY = YY.reshape(xx.shape)
fig, axs = plt.subplots(1, 2, figsize=(15, 6), tight_layout=True)
axs[0].scatter(df['balance'][df['default'] == 1], df['income'][df['default'] == 1], color='red', label = 'Yes')
axs[0].scatter(df['balance'][df['default'] == 0], df['income'][df['default'] == 0], color='blue', alpha=0.5, label = 'No')
axs[0].contour(xx, yy, YY, [0.5], colors='black', linestyles='dashed')
axs[0].set_title('Population - Default')
axs[1].scatter(X['balance'][y == 1],X['income'][y == 1], color='red', label = 'Yes')
axs[1].scatter(X['balance'][y == 0],X['income'][y == 0], color='blue', alpha=0.5, label = 'No')
axs[1].contour(xx, yy, YY, [0.5], colors='black', linestyles='dashed')
axs[1].set_title('Sample - Default')
for ax in axs:
    ax.legend()
    ax.set_ylabel('Income')
    ax.set_xlabel('Balance')
    ax.set_xlim(-100, 3000)
    ax.set_ylim(-1000, 75000)
plt.show()


# Code that generates plots with Linear regression and KNN decision boundaries along with the optimal decision boundary
from matplotlib.colors import ListedColormap
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
lda = LinearDiscriminantAnalysis()
knn = KNeighborsClassifier(n_neighbors=1, n_jobs=-1)
lda.fit(X, y).predict(X)
knn.fit(X, y).predict(X)
cmap_light = ListedColormap(['#AAAAFF', '#FFAAAA'])
fig, axs = plt.subplots(1, 2, figsize=(15, 6), tight_layout=True)
clfs = [lda, knn]
clfs_names = ['Linear Regression', '1NN Classifier']
for i in range(len(clfs)):
    Z = clfs[i].predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    axs[i].pcolormesh(xx, yy, Z, cmap=(cmap_light))
    YY = qda.predict(np.c_[xx.ravel(), yy.ravel()])
    YY = YY.reshape(xx.shape)
    axs[i].set_title(clfs_names[i])
    axs[i].contour(xx, yy, YY, [0.5], colors='black', linestyles='dashed')
    axs[i].scatter(X['balance'][y == 1],X['income'][y == 1], color='red', label = 'Yes')
    axs[i].scatter(X['balance'][y == 0],X['income'][y == 0], color='blue', alpha=0.2, label = 'No')
    axs[i].set_xlabel('Balance')
    axs[i].set_xlabel('Balance')
    plt.show();
