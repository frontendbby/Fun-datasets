import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression

# Generacion de datos aleatorios
np.random.seed(1238)
X = np.arange(1900,2023) # Años de 1923 a 2023

# Tendencia lineal con ruido
y = 850400 + ( X - 2023) * 5 + np.random.normal(0,103, len(X))
print("Longitua del conjunto de datos es: ", len(X))

plt.plot(X, y, 'b.')
plt.xlabel('Años')
plt.ylabel('Coste de vivienda')
plt.show()

data = {'años':X.flatten(), 'coste departamentos':y.flatten()}
df = pd.DataFrame(data)
df.head(10)

lin_reg = LinearRegression()
lin_reg.fit(df['años'].values.reshape(-1,1), df['coste departamentos'].values)

lin_reg.intercept_
lin_reg.coef_

X_min_max = np.array([[df['años'].min()], [df['años'].max()]])
y_train_pred = lin_reg.predict(X_min_max)

plt.plot(X_min_max, y_train_pred, "g-")
plt.plot(df['años'], df['coste departamentos'], "b.")
plt.xlabel('Años')
plt.ylabel('Coste de Departamentos')
plt.show()