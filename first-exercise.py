import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models

# Cargar los datos
dftrain = pd.read_csv('https://storage.googleapis.com/tf-datasets/titanic/train.csv')
dfeval = pd.read_csv('https://storage.googleapis.com/tf-datasets/titanic/eval.csv')

y_train = dftrain.pop('survived')
y_eval = dfeval.pop('survived')

# Convertir todas las columnas categóricas a tipo string
CATEGORICAL_COLUMNS = ['sex', 'n_siblings_spouses', 'parch', 'class', 'deck', 'embark_town', 'alone']
for col in CATEGORICAL_COLUMNS:
    dftrain[col] = dftrain[col].astype(str)
    dfeval[col] = dfeval[col].astype(str)

# Definir las columnas categóricas y numéricas
NUMERIC_COLUMNS = ['age', 'fare']

# Preprocesamiento de columnas categóricas
categorical_inputs = {}
for feature_name in CATEGORICAL_COLUMNS:
    vocabulary = dftrain[feature_name].unique()
    categorical_inputs[feature_name] = layers.StringLookup(vocabulary=vocabulary, output_mode='one_hot')

# Preprocesamiento de columnas numéricas
numeric_inputs = {}
for feature_name in NUMERIC_COLUMNS:
    numeric_inputs[feature_name] = layers.Normalization()
    numeric_inputs[feature_name].adapt(dftrain[feature_name].values)  # Ajustar solo con datos de entrenamiento

# Aplicar el preprocesamiento a los datos de entrenamiento
preprocessed_train = []
for feature_name in CATEGORICAL_COLUMNS:
    preprocessed_train.append(
        tf.cast(categorical_inputs[feature_name](dftrain[feature_name].values), tf.float32)
    )
for feature_name in NUMERIC_COLUMNS:
    preprocessed_train.append(
        tf.expand_dims(
            tf.cast(numeric_inputs[feature_name](dftrain[feature_name].values), tf.float32),
            axis=-1
        )
    )

# Concatenar todas las características preprocesadas
X_train = tf.concat(preprocessed_train, axis=1)

# Crear el modelo
model = models.Sequential([
    layers.Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
    layers.Dense(1, activation='sigmoid')
])

# Compilar el modelo
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Entrenar el modelo
model.fit(X_train, y_train, epochs=10)

# Preprocesar los datos de evaluación
preprocessed_eval = []
for feature_name in CATEGORICAL_COLUMNS:
    preprocessed_eval.append(
        tf.cast(categorical_inputs[feature_name](dfeval[feature_name].values), tf.float32)
    )
for feature_name in NUMERIC_COLUMNS:
    preprocessed_eval.append(
        tf.expand_dims(
            tf.cast(numeric_inputs[feature_name](dfeval[feature_name].values), tf.float32),
            axis=-1
        )
    )

# Concatenar todas las características preprocesadas
X_eval = tf.concat(preprocessed_eval, axis=1)

# Evaluar el modelo
result = model.evaluate(X_eval, y_eval)

print(f"Accuracy: {result[1]}")

# Mostrar graficos

# dftrain.age.hist(bins=20) # muestra el grafico xy
# dftrain.sex.value_counts().plot(kind='barh') # este muestra barras laterales
#pd.concat([dftrain, y_train], axis=1).groupby('sex').survived.mean().plot(kind='barh').set_xlabel('% survive') # este concatena data

# estos son los titutos de los graficos

# plt.xlabel("Edad")
# plt.ylabel("Frecuencia")
# plt.title("Distribución de edades en el conjunto de entrenamiento")

#esto activa la vista de graficos
# plt.show()

#-----------------------------------
# dftrain.head() los 5 primeros items
# dftrain.describe() analisis estadistico de la data
# dftrain.shape muestra el tipo de matriz
# dftrain.loc[0] especifico item
