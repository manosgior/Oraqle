from networks.SingleQubitFNN import SingleQubitFNN

from trainers.SingleQubitFNNTrainer import SingleTrainer
from helpers.data_loader import QubitData
from helpers.config import configuration
from loguru import logger
import json


def main(fnn_definition, qubit_name, trace_length, model):
    fnn_configs = configuration[fnn_definition]
    data_config = fnn_configs['data_config']
    data_config['trace_length'] = trace_length

    nn_config = fnn_configs['nn_config']
    nn_config.update({
        'qubit_name': qubit_name,
        'trace_length': trace_length
    })

    config = fnn_configs['config']
    performance_results_file = config['performance_results_file']

    logger.info(f"Configuration: {config}")
    logger.info(f"NN Configuration: {nn_config}")
    logger.info(f"Data Configuration: {data_config}")

    data_path = config['data_path']
    data_train_file_name = f'DRaw_C_Tr_v0-001_{qubit_name}'
    data_test_file_name = f'DRaw_C_Te_v0-002_{qubit_name}'

    logger.info(f"Training file: {data_train_file_name}, Test file: {data_test_file_name}")

    # Load and transform data
    q = QubitData(data_path, data_train_file_name, data_test_file_name, data_config)
    X_train, y_train, X_val, y_val, X_test, y_test = q.load_transform()

    logger.info(f"Shapes - X_train: {X_train.shape}, y_train: {y_train.shape}")
    logger.info(f"Shapes - X_val: {X_val.shape}, y_val: {y_val.shape}")
    logger.info(f"Shapes - X_test: {X_test.shape}, y_test: {y_test.shape}")

    # Train and evaluate the neural network
    neural_network = SingleTrainer(model, nn_config, config)
    logger.info(f"Neural Network Configuration: {neural_network.__dict__}")

    _, _, best_model_path = neural_network.train(X_train, y_train, X_val, y_val)
    best_model_test_accuracy = neural_network.test_best_model(X_test, y_test, best_model_path)

    logger.info(f"Test Accuracy: {best_model_test_accuracy}")

    del X_train, y_train, X_val, y_val, X_test, y_test

    # Update performance results file
    try:
        with open(performance_results_file, 'r') as json_file:
            performance_results = json.load(json_file)

        performance_results[f"{qubit_name}_trace_d_{trace_length}"] = {
            "best_model_test_accuracy": best_model_test_accuracy,
        }

        with open(performance_results_file, 'w') as json_file:
            json.dump(performance_results, json_file, indent=4)

        logger.info("Performance results updated successfully.")

    except FileNotFoundError:
        logger.error(f"File not found: {performance_results_file}")
    except json.JSONDecodeError:
        logger.error(f"Error decoding JSON from file: {performance_results_file}")
    except Exception as e:
        logger.error(f"An error occurred while updating the performance results: {e}")


if __name__ == "__main__":
    list_single_qubits = ["q1", "q2", "q3", "q4", "q5"]
    trace_durations = [500] #[300, 500, 800, 1000][::-1]

    for qubit in list_single_qubits:
        _fnn_definition = "SingleQubitFNN"
        _qubit_name = qubit
        for _trace_length in trace_durations:
            _input_size = 2 * _trace_length
            _model = SingleQubitFNN(_input_size, 1)
            main(_fnn_definition, _qubit_name, _trace_length, _model)
