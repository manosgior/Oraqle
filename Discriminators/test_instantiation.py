import sys
import torch

from networks import (
    Arxiv240618807FNN,
    Net, Net_rmf,
    CNN,
    QubitClassifierTransformer,
    SingleQubitFNN, SingleQubitFNN_Baseline, SingleQubitFNN_StudentModel,
    KLiNQTeacherModel, KLiNQStudentModel,
    get_model_info
)

def test_models():
    print("Testing Arxiv240618807FNN...")
    model = Arxiv240618807FNN()
    get_model_info(model)

    print("\nTesting HERQULES Net...")
    model = Net()
    get_model_info(model)
    
    print("\nTesting HERQULES Net_rmf...")
    model = Net_rmf()
    get_model_info(model)

    print("\nTesting CNN...")
    model = CNN()
    get_model_info(model)

    print("\nTesting Transformer...")
    model = QubitClassifierTransformer()
    get_model_info(model)

    print("\nTesting SingleQubitFNN...")
    model = SingleQubitFNN(1000, 32)
    get_model_info(model)

    print("\nTesting SingleQubitFNN_Baseline...")
    model = SingleQubitFNN_Baseline()
    get_model_info(model)

    print("\nTesting SingleQubitFNN_StudentModel...")
    model = SingleQubitFNN_StudentModel(1000, 32)
    get_model_info(model)

    print("\nTesting KLiNQTeacherModel...")
    model = KLiNQTeacherModel(1000, 32)
    get_model_info(model)

    print("\nTesting KLiNQStudentModel...")
    model = KLiNQStudentModel(31)
    get_model_info(model)

if __name__ == "__main__":
    test_models()
    print("All models instantiated successfully.")
