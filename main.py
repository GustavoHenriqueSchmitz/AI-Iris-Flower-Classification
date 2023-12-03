import tensorflow as tf
from tensorflow import keras
import pandas as pd

# Getting the data
train_data = pd.read_csv('./IRIS_Train.csv')
test_data = pd.read_csv('./IRIS_Test.csv')

# Preprocessing the data for train
x_train = train_data.drop('species', axis=1) # Separate the necessary data for the AI analyse and classify
y_train = pd.get_dummies(train_data.species) # Convert categorical variable into dummy/indicator variables

# Preprocessing the data for test
x_test = test_data.drop('species', axis=1) # Separate the necessary data for the AI analyse and classify
y_test = pd.get_dummies(test_data.species) # Convert categorical variable into dummy/indicator variables

# Create the model
model = keras.Sequential([
    keras.layers.Dense(64, activation='relu', input_shape=[len(x_train.keys())]),
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dense(3, activation='softmax')
])

# Compile the model
model.compile(
    loss='categorical_crossentropy', # Loss function
    optimizer='adam', # Optimizer
    metrics=['accuracy'] # Metric to monitor
)

# Train the model
history = model.fit(x_train, y_train, epochs=300, validation_split = 0.2)

# Evaluate the model on the test data
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)

# Print the test results
print("================= Test Results ==================")
print(f"Loss: {test_loss} | Accurate: {test_acc}")
print("=================================================")

# Make a prediction on a new data point
predictions = model.predict(pd.DataFrame([[5.3, 3.7, 1.5, 0.2]]))
for counter, prediction in enumerate(predictions[0]):
    if counter == 0:
        print(f"Iris-setosa: {prediction * 100}")
    elif counter == 1:
        print(f"Iris-versicolor: {prediction * 100}")
    else:
        print(f"Iris-virginica: {prediction * 100}")
