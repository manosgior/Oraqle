from pathlib import Path

current_dir = Path(__file__).parent
root_dir = Path("/Users/tbunarjyan/Desktop/KLiNQ/KLiNQ/quibit-readout-klinq")
root_dir = Path("/Users/tbunarjyan/Desktop/KLiNQ/KLiNQ/quibit-readout-klinq")

configuration = {
    'SingleQubitFNN': {
        'config': {
            'data_path': root_dir.joinpath('data').joinpath('single_qubit_data'),
            'model_storage_path': root_dir.joinpath('artifacts').joinpath('original_models').joinpath('SingleQubitFNN'),
            'performance_results_file': root_dir.joinpath('artifacts').joinpath('performance').joinpath(
                'SingleQubitFNN.json'),
        },
        'data_config': {
            'train_sample_size': 216000,
            'val_sample_size': 24000,
            'normalize': 'mean/std', # "no-norm" etc.
            'trace_length': '',  # is set as input later
            'val_sampling_mode': 'random'  # stratifiedls
        },
        'nn_config': {
            'trace_length': '',  # is set as input later
            'num_epochs': 300,
            'batch_size': 1024,
            'learning_rate': 0.0001,
            'fine_tune_learning_rate': 0.00001,
            'fine_tune_num_epochs': 5,
            'device': 'mps',
            'patience': 15,
            'delta': 0
        }
    },
    'SingleQubitStudentModel': {
        'config': {
            'data_path': root_dir.joinpath('data').joinpath('single_qubit_data'),
            'model_storage_path': root_dir.joinpath('artifacts').joinpath('distilled_models').joinpath(
                'SingleQubitStudentModel'),
            'performance_results_file': root_dir.joinpath('artifacts').joinpath('performance').joinpath('SingleQubitStudentModel.json'),

        },
        'data_config': {
            'train_sample_size': 216000,
            'val_sample_size': 24000,
            'normalize': 'mean/std',
            'trace_length': '',  # is set as input later
            'val_sampling_mode': 'random'  # stratified
        },
        'nn_config': {
            'trace_length': '',  # is set as input later
            'num_epochs': 30,
            'batch_size': 1024,
            'learning_rate': 0.0001,
            'fine_tune_learning_rate': 0.0001,
            'fine_tune_num_epochs': 3,
            'device': 'cpu',
            'patience': 5,
            'delta': 0.0001,
            'distillation_settings': {
                'T4-ST2-CE8': {
                    'temperature': 4,
                    'soft_target_loss_weight': 0.2,
                    'ce_loss_weight': 0.8
                },
                'T2-ST3-CE7': {
                    'temperature': 3,
                    'soft_target_loss_weight': 0.3,
                    'ce_loss_weight': 0.7
                },
                'T5-ST5-CE5': {
                    'temperature': 5,
                    'soft_target_loss_weight': 0.5,
                    'ce_loss_weight': 0.5
                }
            }
        }
    },

    'KLiNQTeacherModel': {
        'config': {
            'data_path': root_dir.joinpath('data').joinpath('single_qubit_data'),
            'model_storage_path': root_dir.joinpath('artifacts').joinpath('original_models').joinpath(
                'KLiNQTeacherModel'),
            'performance_results_file': root_dir.joinpath('artifacts').joinpath('performance').joinpath(
                'KLiNQTeacherModel.json'),
        },
        'data_config': {
            'train_sample_size': 216000,
            'val_sample_size': 24000,
            'normalize': 'mean/std',
            'trace_length': '',  # is set as input later
            'val_sampling_mode': 'random'  # stratified
        },
        'nn_config': {
            'trace_length': '',  # is set as input later
            'num_epochs': 300,
            'batch_size': 1024,
            'learning_rate': 0.001,
            'fine_tune_learning_rate': 0.00001,
            'fine_tune_num_epochs': 5,
            'device': 'mps',
            'patience': 15,
            'delta': 0
        }
    },
    'KLiNQStudentModel': {
        'config': {
            'data_path': root_dir.joinpath('data').joinpath('single_qubit_data'),
            'model_storage_path': root_dir.joinpath('artifacts').joinpath('distilled_models').joinpath(
                'KLiNQStudentModel'),
            'performance_results_file': root_dir.joinpath('artifacts').joinpath('performance').joinpath(
                'KLiNQStudentModel.json'),
        },
        'data_config': {
            'train_sample_size': 216000,
            'val_sample_size': 24000,
            'normalize': 'mean/std',
            'trace_length': '',  # is set as input later
            'val_sampling_mode': 'random'  # stratified
        },
        'nn_config': {
            'trace_length': '',  # is set as input later
            'num_epochs': 50,
            'batch_size': 1024,
            'learning_rate': 0.001,
            'fine_tune_learning_rate': 0.0001,
            'fine_tune_num_epochs': 3,
            'device': 'mps',
            'patience': 5,
            'delta': 0.0001,
            'distillation_settings': {
                'T4-ST2-CE8': {
                    'temperature': 4,
                    'soft_target_loss_weight': 0.2,
                    'ce_loss_weight': 0.8
                },
                'T2-ST3-CE7': {
                    'temperature': 3,
                    'soft_target_loss_weight': 0.3,
                    'ce_loss_weight': 0.7
                },
                'T5-ST5-CE5': {
                    'temperature': 5,
                    'soft_target_loss_weight': 0.5,
                    'ce_loss_weight': 0.5
                },
                'T4-ST8-CE2': {
                    'temperature': 5,
                    'soft_target_loss_weight': 0.8,
                    'ce_loss_weight': 0.2
                },
                'T2-ST7-CE3': {
                    'temperature': 2,
                    'soft_target_loss_weight': 0.7,
                    'ce_loss_weight': 0.3
                },
                'T4-ST3-CE7': {
                    'temperature': 5,
                    'soft_target_loss_weight': 0.3,
                    'ce_loss_weight': 0.7
                },
                'T4-ST9-CE1': {
                    'temperature': 4,
                    'soft_target_loss_weight': 0.9,
                    'ce_loss_weight': 0.1
                },
                'T1-ST9-CE1': {
                    'temperature': 1,
                    'soft_target_loss_weight': 0.9,
                    'ce_loss_weight': 0.1
                },
            }
        }
    },
}
