from tensorflow import keras
import pandas as pd

# Load the model for use
loaded_model = keras.models.load_model('IRIS_Classification_model.h5')

print("======================================================")
flower_size_values = []
for size_type in ["sepal_length", "sepal_width", "petal_length", "petal_width"]:
    flower_size_values.append(float(input(f"{size_type}: ")))

# Make a prediction on a new data point
predictions = loaded_model.predict(pd.DataFrame([flower_size_values]))
print("======================================================")
print("Probability for each species: ")
print("======================================================")
for counter, prediction in enumerate(predictions[0]):
    if counter == 0:
        print(f"Iris-setosa: {prediction * 100}%")
    elif counter == 1:
        print(f"Iris-versicolor: {prediction * 100}%")
    else:
        print(f"Iris-virginica: {prediction * 100}%")
