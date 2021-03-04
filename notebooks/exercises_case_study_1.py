# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: hydrogen
#       format_version: '1.3'
#       jupytext_version: 1.9.1
#   kernelspec:
#     display_name: Python 3
#     name: python3
# ---

# %% id="t002L7LL9fnv"
# Solution 1
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import numbers


path_to_data = 'https://raw.githubusercontent.com/armoutihansen/EEMP2020/main/datasets/AMP_Data.csv'
df = pd.read_csv(path_to_data)

# %% colab={"base_uri": "https://localhost:8080/"} id="HfKA6zcY9ol0" outputId="0adf37ae-0647-484e-e6b8-5467faac771e"
# Solution 2

cols = ['roce','lean1','lean2','perf1','perf2','perf3','perf4',
        'perf5','perf6','perf7','perf8','perf9','perf10','talent1','talent2',
        'talent3','talent4','talent5','talent6']

df = df[cols]

df = df.dropna()

print(df.info())

# %% id="XW0r2RhN90ZC"
# Solution 3
cols.remove('roce')

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(df[cols],
                                                    df['roce'], train_size=0.75, random_state=181)

# %% colab={"base_uri": "https://localhost:8080/", "height": 1000} id="1Fi5LQa9-D_W" outputId="e929b520-e9d2-4307-f239-eb9cdaaf8885"
# Solution 4
from sklearn.linear_model import LinearRegression

reg = LinearRegression().fit(X_train, y_train)

pred = reg.predict(X_test)

from sklearn.metrics import r2_score

print('R2:', r2_score(y_test, pred))

print(pd.DataFrame({'feature': X_train.columns, 'coef:': reg.coef_}))

plt.figure(figsize=(20,20))
(pd.Series(reg.coef_, index=X_train.columns).nlargest(18).plot(kind='barh')) 
plt.xlabel('Coef')
plt.ylabel('Feature')
plt.show;

# %% colab={"base_uri": "https://localhost:8080/", "height": 406} id="Slft6D5I-ntL" outputId="d8e67222-9d4f-4f15-b072-528fb0b4b517"
# Solution 5
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor

param_grid = {'n_estimators': np.arange(500,1001,100),
              'max_features': np.arange(1,19),}
ran_grid = RandomizedSearchCV(RandomForestRegressor(random_state=181),
                        param_grid, cv=5, scoring='neg_mean_squared_error', n_iter=10, verbose=1, n_jobs=-1)

ran_grid.fit(X_train, y_train)

pred = ran_grid.predict(X_test)

print('R2: ', r2_score(y_test, pred))

plt.figure(figsize=(20,7))
(pd.Series(ran_grid.best_estimator_.feature_importances_*100, index=X_train.columns).nlargest(8).plot(kind='barh')) 
plt.xlabel('Percentage improvement')
plt.ylabel('Feature')
plt.title('Feature Importance of Random Forest Regression')
plt.show;

# %% colab={"base_uri": "https://localhost:8080/", "height": 296} id="0e0TSrTJ_TbY" outputId="4c2340b6-a585-48f8-ffdc-0f31c7f61810"
# Solution 6
from sklearn.inspection import plot_partial_dependence
plot_partial_dependence(ran_grid, X_train, ['talent6', 'talent2'])
