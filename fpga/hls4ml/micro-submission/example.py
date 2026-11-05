import hls4ml
import numpy as np

from keras.models import Sequential
from keras.layers import Dense, Activation


# Construct a basic keras model
model = Sequential()
model.add(Dense(64, input_shape=(16,), activation='relu'))
model.add(Dense(32, activation='relu'))

# This is where you would train the model in a real-world scenario

# Generate an hls configuration from the keras model
config = hls4ml.utils.config_from_keras_model(model)

# You can print the config to see some default parameters
print(config)

# Convert the model to an hls project using the config
hls_model = hls4ml.converters.convert_from_keras_model(
    model=model,
    hls_config=config,
    backend='Vitis'
)

# Compile the hls project and link it into the Python runtime
hls_model.compile()

# Generate random input data
X_input = np.random.rand(100, 16)

# Run the model on the input data
hls_prediction = hls_model.predict(X_input)

# Use Vitis HLS to synthesize the model
# This might take several minutes
hls_model.build()

# Optional: print out the report
hls4ml.report.read_vivado_report('my-hls-test')
