import tensorflow as tf
import numpy as np

# Generate fake training data (replace with real sensor data)
X_train = np.random.rand(1000, 4)  # Inputs: Random sensor data
Y_train = np.random.rand(1000, 12) # Outputs: Servo angles for 12 joints

# Define the AI model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(16, activation="relu", input_shape=(4,)),  # Input: Sensor readings
    tf.keras.layers.Dense(32, activation="relu"),
    tf.keras.layers.Dense(12, activation="sigmoid")  # Output: 12 servo angles
])

# Compile & Train
model.compile(optimizer="adam", loss="mse")
model.fit(X_train, Y_train, epochs=50)

# Convert to TensorFlow Lite model
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

# Save model
with open("quadruped_model.tflite", "wb") as f:
    f.write(tflite_model)
