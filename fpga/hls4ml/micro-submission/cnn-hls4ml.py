import hls4ml
import keras

# The model is saved by cnn-model.py
model = keras.models.load_model('cnn-model.keras')

# Generate an hls configuration from the keras model
config = hls4ml.utils.config_from_keras_model(
    model,
    granularity='name',
    default_precision='ap_int<8>'
)
config['Model']['ClockPeriod'] = 2  # 2 ns = 500 MHz
print(config)

# Convert the model to an hls project using the config
hls_model = hls4ml.converters.convert_from_keras_model(
    model=model,
    hls_config=config,
    backend='Vitis',
    part='xcu280-fsvh2892-2l-e'
)

# Save model to disk
hls_model.save('cnn-model.fml')

# Compile the hls project and link it into the Python runtime
hls_model.compile()

# Generate random input data
# X_input = np.random.rand(100, 16)

# Run the model on the input data
# hls_prediction = hls_model.predict(X_input)

# Use Vitis HLS to synthesize the model
# This might take several minutes
hls_model.build()

# Optional: print out the report
hls4ml.report.read_vivado_report('my-hls-test')
