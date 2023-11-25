import tensorflow as tf
from tensorflow import keras
import pandas as pd

# Getting the data
train_data = pd.read_csv('./IRIS_Train.csv')
test_data = pd.read_csv('./IRIS_Test.csv')

# Preprocessing the data for train
X_train = train_data.drop('species', axis=1) # Separate the necessary data for the AI analyse and classify
y_train = pd.get_dummies(train_data.species) # Convert categorical variable into dummy/indicator variables

# Preprocessing the data for test
X_test = test_data.drop('species', axis=1) # Separate the necessary data for the AI analyse and classify
y_test = pd.get_dummies(test_data.species) # Convert categorical variable into dummy/indicator variables

# Create the model
model = keras.Sequential([
    keras.layers.Dense(64, activation='relu', input_shape=[len(X_train.keys())]),
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dense(3, activation='softmax')
])

model.compile(
    loss='categorical_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)

history = model.fit(X_train, y_train, epochs=1000, validation_split = 0.2)

test_loss, test_acc = model.evaluate(X_test, y_test, verbose=2)
