from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import clear_output
from six.moves import urllib

import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
import tensorflow as tf
import tensorflow.compat.v2.feature_column as fc


dftrain = pd.read_csv('https://storage.googleapis.com/tf-datasets/titanic/train.csv')
dfeval = pd.read_csv('https://storage.googleapis.com/tf-datasets/titanic/eval.csv')



y_train = dftrain.pop('survived')
y_eval = dfeval.pop('survived')

# Mostrar graficos

# dftrain.age.hist(bins=20) # muestra el grafico xy
# dftrain.sex.value_counts().plot(kind='barh') # este muestra barras laterales
#pd.concat([dftrain, y_train], axis=1).groupby('sex').survived.mean().plot(kind='barh').set_xlabel('% survive') # este concatena data

# estos son los titutos de los graficos

# plt.xlabel("Edad")
# plt.ylabel("Frecuencia")
# plt.title("Distribución de edades en el conjunto de entrenamiento")

#esto activa la vista de graficos
plt.show()

#print(dftrain.shape)

# dftrain.head() los 5 primeros items
# dftrain.describe() analisis estadistico de la data
# dftrain.shape muestra el tipo de matriz
# dftrain.loc[0] especifico item

