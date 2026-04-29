from networks.SingleQubitFNN import SingleQubitFNN
from trainers.KnowledgeDistillationTrainer_SingleQubitFNN import KnowledgeDistillationSingleQubitFNN

from networks.SingleQubitFNN_StudentModel import SingleQubitStudentModel

from helpers.data_loader import QubitData
from helpers.config import configuration
from loguru import logger
import json
import torch

def knowledge_distillation(teacher_model,
                           student_model,
                           fnn_definition,
                           student_fnn_definition,
                           qubit_name,
                           trace_length,
                           best_teacher_model_path,
                           knowledge_distillation_setting):
    fnn_configs = configuration[student_fnn_definition]  #### changed to student_fnn_definition
    data_config = fnn_configs['data_config']
    data_config['trace_length'] = trace_length

    nn_config = fnn_configs['nn_config']
    nn_config.update({
        'qubit_name': qubit_name,
        'trace_length': trace_length
    })

    config = fnn_configs['config']
    kd_performance_results_file = config['performance_results_file']

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

    neural_network = KnowledgeDistillationSingleQubitFNN(teacher_model, student_model,
                                                         nn_config, config,
                                                         knowledge_distillation_setting)
    logger.info(f"Neural Network Configuration: {neural_network.__dict__}")

    best_teacher_test_accuracy, best_teacher_model_outcomes = neural_network.test_best_teacher_model(
        X_test, y_test, best_teacher_model_path, 2*trace_length)

    student_best_model_path = neural_network.train_knowledge_distillation(
        best_teacher_model_path, X_train,y_train, X_val, y_val, trace_length=2*trace_length)

    best_student_test_accuracy, best_student_model_outcomes = neural_network.test_best_student_model(
        X_test, y_test, student_best_model_path)

    logger.info(f"Best Teacher Test Accuracy: {best_teacher_test_accuracy}")
    logger.info(f"Best Student Test Accuracy: {best_student_test_accuracy}")

    del X_train, y_train, X_val, y_val, X_test, y_test

    try:
        with open(kd_performance_results_file, 'r') as json_file:
            kd_performance_results = json.load(json_file)

        if knowledge_distillation_setting not in kd_performance_results:
            kd_performance_results[knowledge_distillation_setting] = {}

        kd_performance_results[knowledge_distillation_setting][f"{qubit_name}_trace_d_{trace_length}"] = {
            "best_teacher_model_test_accuracy": best_teacher_test_accuracy,
            "best_teacher_model_num_non_zero_params": best_teacher_model_outcomes['num_non_zero_params'],
            "best_teacher_model_num_params": best_teacher_model_outcomes['num_total_params'],
            "best_student_model_test_accuracy": best_student_test_accuracy,
            "best_student_model_num_non_zero_params": best_student_model_outcomes['num_non_zero_params'],
            "best_student_model_num_params": best_student_model_outcomes['num_total_params']
        }

        with open(kd_performance_results_file, 'w') as json_file:
            json.dump(kd_performance_results, json_file, indent=4)

        logger.info("Performance results updated successfully.")

    except FileNotFoundError:
        logger.error(f"File not found: {kd_performance_results_file}")
    except json.JSONDecodeError:
        logger.error(f"Error decoding JSON from file: {kd_performance_results_file}")
    except Exception as e:
        logger.error(f"An error occurred while updating the performance results: {e}")


if __name__ == "__main__":

    list_single_qubits = ["q1", "q2", "q3", "q4", "q5"]
    trace_durations = [500] # [300, 500, 800, 1000]
    knowledge_distillation_setting = ['T4-ST2-CE8']  #'T2-ST3-CE7', 'T5-ST5-CE5'

    for _trace_length in trace_durations:
        _fnn_definition = "SingleQubitFNN"
        _student_fnn_definition = "SingleQubitStudentModel"
        _input_size = 2 * _trace_length
        torch.manual_seed(42)
        _teacher_model = SingleQubitFNN(_input_size, 1)
        torch.manual_seed(42)
        _student_model = SingleQubitStudentModel(_input_size, 1)

        for _qubit_name in list_single_qubits:
            _best_teacher_model_path = (f"{configuration[_fnn_definition]['config']['model_storage_path']}"
                                        f"/{_qubit_name}_trace_{_trace_length}_best_model.pth")
            print(_best_teacher_model_path)

            for _knowledge_distillation_setting in knowledge_distillation_setting:
                knowledge_distillation(_teacher_model, _student_model,
                                       _fnn_definition, _student_fnn_definition,
                                       _qubit_name, _trace_length,
                                       _best_teacher_model_path,
                                       _knowledge_distillation_setting)