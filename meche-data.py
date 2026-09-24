#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


# In[5]:


df = pd.read_csv('mechanical.csv')
df.head()


# In[6]:


# Drop record_id column
df = df.drop(columns=['record_id'])

# preprocess the data
df = df.dropna()

df.head()


# In[7]:


# normalize the data
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
df[['rotational_speed_rpm', 'load_kN', 'bearing_temperature_c', 'lubricant_level_pct', 'vibration_mm_s' ]] = scaler.fit_transform(df[['rotational_speed_rpm', 'load_kN', 'bearing_temperature_c', 'lubricant_level_pct', 'vibration_mm_s']])


# In[8]:


# split the data into training and testing sets
from sklearn.model_selection import train_test_split
X = df.drop(columns=['condition_class'])
y = df['condition_class']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)


# In[9]:


# save the training and testing sets to csv files for later use
X_train.to_csv('X_train.csv', index=False)
X_test.to_csv('X_test.csv', index=False)
y_train.to_csv('y_train.csv', index=False)
y_test.to_csv('y_test.csv', index=False)


# In[10]:


# load the saved datasets using pd.read_csv('X_train.csv') etc.
X_train_loaded = pd.read_csv('X_train.csv')
X_test_loaded = pd.read_csv('X_test.csv')
y_train_loaded = pd.read_csv('y_train.csv')
y_test_loaded = pd.read_csv('y_test.csv')


# In[11]:


# check if the loaded datasets are the same as the original datasets
print(
    "X_train matches:",
    np.allclose(
        X_train.reset_index(drop=True).to_numpy(),
        X_train_loaded.to_numpy()
    )
)

print(
    "X_test matches:",
    np.allclose(
        X_test.reset_index(drop=True).to_numpy(),
        X_test_loaded.to_numpy()
    )
)

print(
    "y_train matches:",
    y_train.reset_index(drop=True).equals(
        y_train_loaded.squeeze("columns")
    )
)

print(
    "y_test matches:",
    y_test.reset_index(drop=True).equals(
        y_test_loaded.squeeze("columns")
    )
)


# In[12]:


# build and train a neural network model
tf.random.set_seed(42)

model = keras.Sequential([
    layers.Input(shape=(X_train.shape[1],)),
    layers.Dense(32, activation="relu"),
    layers.Dropout(0.2),
    layers.Dense(16, activation="relu"),
    layers.Dense(3, activation="softmax")
])

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

early_stopping = keras.callbacks.EarlyStopping(
    monitor="val_loss",
    patience=15,
    restore_best_weights=True
)

history = model.fit(
    X_train,
    y_train,
    validation_split=0.2,
    epochs=200,
    batch_size=8,
    callbacks=[early_stopping],
    verbose=1
)

test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
print(f"Test loss: {test_loss:.4f}")
print(f"Test accuracy: {test_accuracy:.4f}")

predicted_classes = model.predict(X_test).argmax(axis=1)
print("Predicted classes:", predicted_classes)


# In[13]:


# Classification performance plots.
# The classifier has three output probabilities and uses categorical loss.
classification_history = history.history

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].plot(classification_history["loss"], label="Training loss")
axes[0].plot(classification_history["val_loss"], label="Validation loss")
axes[0].set_title("Classification loss")
axes[0].set_xlabel("Epoch")
axes[0].set_ylabel("Sparse categorical cross-entropy")
axes[0].legend()
axes[1].plot(classification_history["accuracy"], label="Training accuracy")
axes[1].plot(classification_history["val_accuracy"], label="Validation accuracy")
axes[1].set_title("Classification accuracy")
axes[1].set_xlabel("Epoch")
axes[1].set_ylabel("Accuracy")
axes[1].legend()
plt.tight_layout()
plt.show()


# In[14]:


# REGRESSION MODEL
# This section intentionally uses the same condition_class values as a numeric
# target. It demonstrates regression, but condition_class is fundamentally a
# categorical label, so classification is the better model for the actual task.
# Unlike the classifier, this model has one linear output and predicts values
# such as 0.2 or 1.8 rather than class probabilities.
tf.random.set_seed(42)

regression_model = keras.Sequential([
    layers.Input(shape=(X_train.shape[1],)),
    layers.Dense(32, activation="relu"),
    layers.Dropout(0.2),
    layers.Dense(16, activation="relu"),
    layers.Dense(1, activation="linear")
])

regression_model.compile(
    optimizer="adam",
    loss="mse",
    metrics=[keras.metrics.MeanAbsoluteError(name="mae")]
)

regression_early_stopping = keras.callbacks.EarlyStopping(
    monitor="val_loss",
    patience=15,
    restore_best_weights=True
)

regression_history = regression_model.fit(
    X_train,
    y_train.astype("float32"),
    validation_split=0.2,
    epochs=200,
    batch_size=8,
    callbacks=[regression_early_stopping],
    verbose=1
)

regression_loss, regression_mae = regression_model.evaluate(
    X_test,
    y_test.astype("float32"),
    verbose=0
)
regression_rmse = np.sqrt(regression_loss)
print(f"Regression test MSE: {regression_loss:.4f}")
print(f"Regression test RMSE: {regression_rmse:.4f}")
print(f"Regression test MAE: {regression_mae:.4f}")

regression_predictions = regression_model.predict(X_test, verbose=0).ravel()
regression_results = pd.DataFrame({
    "actual_class": y_test.to_numpy(),
    "predicted_numeric_value": regression_predictions,
    "predicted_nearest_class": np.clip(
        np.rint(regression_predictions), 0, 2
    ).astype(int)
})
print(regression_results.head())


# Regression performance plots.
regression_history_data = regression_history.history

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].plot(regression_history_data["loss"], label="Training loss")
axes[0].plot(regression_history_data["val_loss"], label="Validation loss")
axes[0].set_title("Regression loss")
axes[0].set_xlabel("Epoch")
axes[0].set_ylabel("Mean squared error")
axes[0].legend()
axes[1].plot(regression_history_data["mae"], label="Training MAE")
axes[1].plot(regression_history_data["val_mae"], label="Validation MAE")
axes[1].set_title("Regression mean absolute error")
axes[1].set_xlabel("Epoch")
axes[1].set_ylabel("MAE")
axes[1].legend()
plt.tight_layout()
plt.show()

