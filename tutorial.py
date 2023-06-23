import tensorflow as tf
import pandas as pd
import numpy as np
import seaborn as sns 
import matplotlib.pyplot as plt 


#Importació de dades 
temperature_df = pd.read_csv("celsius_a_fahrenheit.csv")


#Visualització
sns.scatterplot(data=temperature_df, x='Celsius', y='Fahrenheit')
plt.show()

#Carrega De Dades
x_train = temperature_df['Celsius']
y_train = temperature_df['Fahrenheit']
#Crear Modul
model = tf.keras.Sequential()
model.add(tf.keras.layers.Dense(units=1, input_shape=[1]))

#Compilació
model.compile(optimizer=tf.keras.optimizers.Adam(1), loss='mean_squared_error')

#Entrenament del model
epochs_hist = model.fit(x_train, y_train, epochs=100)

#Evaluació
epochs_hist.history.keys()

#Gràfic 
plt.plot(epochs_hist.history['loss'])
plt.title("Progreso Perdida durant Entrenament")
plt.xlabel("Epoch")
plt.ylabel("Perdida")
plt.legend("Perdida")
#plt.show()

model.get_weights()

#Prediccions
Temp_C = 0
Temp_F = model.predict([Temp_C])
print(Temp_F)

Temp_F = 9/5 * Temp_C + 32
print(Temp_F)
