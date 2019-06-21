from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np
import pandas as pd
import tensorflow as tf

from tensorflow import feature_column
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split

print("Version: ", tf.__version__)
print("Eager mode: ", tf.executing_eagerly())

# download and prepare the dataset
URL = "./datasets/heart.csv"
dataframe = pd.read_csv(URL)
dataframe.head()
train, test = train_test_split(dataframe, test_size=0.2)
train, val = train_test_split(train, test_size=0.2)
print(len(train), "train examples")
print(len(val), "validation examples")
print(len(test), "test examples")

# A utility method to create a tf.data dataset from a Pandas Dataframe
def df_to_dataset(dataframe, shuffle=True, batch_size=32):
    dataframe = dataframe.copy()
    labels = dataframe.pop("target")
    ds = tf.data.Dataset.from_tensor_slices((dict(dataframe), labels))
    if shuffle:
        ds = ds.shuffle(buffer_size=len(dataframe))
    ds = ds.batch(batch_size)
    return ds


# prepare the dataset (these values are not normalized)
batch_size = 32
train_ds = df_to_dataset(train, batch_size=batch_size)
val_ds = df_to_dataset(val, shuffle=False, batch_size=batch_size)
test_ds = df_to_dataset(test, shuffle=False, batch_size=batch_size)

# turn (structured) dataset into categorized and normalized feature_columns
feature_columns = []

# numeric cols
for header in ["age", "trestbps", "chol", "thalach", "oldpeak", "slope", "ca"]:
    feature_columns.append(feature_column.numeric_column(header))

# bucketized cols
age = feature_column.numeric_column("age")
age_buckets = feature_column.bucketized_column(
    age, boundaries=[18, 25, 30, 35, 40, 45, 50, 55, 60, 65]
)
feature_columns.append(age_buckets)

# indicator cols
thal = feature_column.categorical_column_with_vocabulary_list(
    "thal", ["fixed", "normal", "reversible"], num_oov_buckets=2
)
thal_one_hot = feature_column.indicator_column(thal)
feature_columns.append(thal_one_hot)

# embedding cols
thal_embedding = feature_column.embedding_column(thal, dimension=8)
feature_columns.append(thal_embedding)

# crossed cols
crossed_feature = feature_column.crossed_column(
    [age_buckets, thal], hash_bucket_size=1000
)
crossed_feature = feature_column.indicator_column(crossed_feature)
feature_columns.append(crossed_feature)

# compile the model
model = tf.keras.Sequential(
    [
        tf.keras.layers.DenseFeatures(
            feature_columns  # turn feature columns into keras layer
        ),
        layers.Dense(128, activation="relu"),
        layers.Dense(128, activation="relu"),
        layers.Dense(1, activation="sigmoid"),
    ]
)

model.compile(
    optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"], run_eagerly=True
)

model.fit(train_ds, validation_data=val_ds, epochs=100)

loss, accuracy = model.evaluate(test_ds)
print("Accuracy", accuracy)

row_data = {
    "age": [60],
    "sex": [1],
    "cp": [4],
    "trestbps": [130],
    "chol": [206],
    "fbs": [0],
    "restecg": [2],
    "thalach": [132],
    "exang": [1],
    "oldpeak": [2.4],
    "slope": [2],
    "ca": [2],
    "thal": ["reversible"],
    "target": [-1],
}
df = pd.DataFrame(data=row_data)
n_input = df_to_dataset(df, False, 1)
print(model.predict(n_input))
print("Should print 1")
