import time
import torch
import torch.nn as nn
import torch.optim as optim
from helpers.nn_utils import get_data_loader, get_device


def count_nonzero_parameters(model):
    return sum(torch.count_nonzero(p).item() for p in model.parameters() if p.requires_grad)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class KnowledgeDistillationSingleQubitFNN:
    def __init__(self, teacher_model, student_model, nn_config, config, distillation_setting):
        self.trace_length = nn_config.get('trace_length', None)
        self.qubit_name = nn_config['qubit_name']

        self.num_epochs = nn_config.get('num_epochs', 300)
        self.batch_size = nn_config.get('batch_size', 512)
        self.learning_rate = nn_config.get('learning_rate', 0.001)
        self.patience = nn_config.get('patience', 10)
        self.delta = nn_config.get('delta', 0)

        self.device_type = nn_config.get('device', 'cpu')
        self.device = get_device(self.device_type)
        self.teacher_model = teacher_model.to(self.device)
        self.student_model = student_model.to(self.device)

        self.distillation_setting = distillation_setting
        distillation_settings = nn_config.get('distillation_settings', {}).get(distillation_setting, {})
        self.soft_target_loss_weight = float(distillation_settings.get('soft_target_loss_weight', 0.5))
        self.ce_loss_weight = float(distillation_settings.get('ce_loss_weight', 0.5))
        self.temperature = float(distillation_settings.get('temperature', 2))

        self.model_save_directory = config.get('model_storage_path', './')

    def validate(self, loader, model, criterion):
        model.eval()
        total_loss = 0
        n_correct = 0
        n_samples = 0

        with torch.no_grad():
            for inputs, labels in loader:
                inputs = inputs.to(self.device)
                labels = labels.to(self.device).float().unsqueeze(1)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                total_loss += loss.item()

                probabilities = torch.sigmoid(outputs)
                predicted = (probabilities > 0.5).float()
                n_samples += labels.size(0)
                n_correct += (predicted == labels).sum().item()

        avg_loss = total_loss / len(loader)
        accuracy = 100.0 * n_correct / n_samples

        return avg_loss, accuracy

    def loss_optimizer(self, model):
        criterion = nn.BCEWithLogitsLoss().to(self.device)
        optimizer = optim.Adam(model.parameters(), lr=self.learning_rate)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.75,
            patience=self.patience,
            cooldown=2
        )
        return criterion, optimizer, scheduler

    def train_knowledge_distillation(self, best_teacher_model_path, X_train, y_train, X_val, y_val, trace_length):
        # Load the best teacher model weights
        weights = torch.load(best_teacher_model_path)
        self.teacher_model.load_state_dict(weights)

        # Initialize loss functions and optimizer
        criterion = nn.BCEWithLogitsLoss().to(self.device)  # For classification loss
        distillation_criterion = nn.MSELoss().to(self.device)  # Using MSELoss for distillation
        optimizer = optim.Adam(self.student_model.parameters(), lr=self.learning_rate)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.75,
            patience=self.patience,
            cooldown=2
        )

        train_loader = get_data_loader(X_train, y_train, self.batch_size)
        val_loader = get_data_loader(X_val, y_val, self.batch_size)

        best_val_loss = None
        counter = 0
        training_start_time = time.time()

        self.teacher_model.eval()  # Teacher set to evaluation mode
        self.student_model.train()  # Student set to training mode

        student_best_model_path = None  # Initialize to None

        for epoch in range(self.num_epochs):
            running_loss = 0.0
            n_correct = 0
            n_samples = 0

            for inputs, labels in train_loader:
                inputs = inputs.to(self.device)
                labels = labels.to(self.device).float().unsqueeze(1)

                optimizer.zero_grad()

                teacher_inputs = inputs
                student_inputs = inputs

                # Teacher predictions (with no gradient computation)
                with torch.no_grad():
                    teacher_logits = self.teacher_model(teacher_inputs) / self.temperature  # Apply temperature scaling

                # Student predictions
                student_logits = self.student_model(student_inputs) / self.temperature  # Apply temperature scaling

                # Distillation loss using MSE between logits
                distillation_loss = distillation_criterion(student_logits, teacher_logits) * (self.temperature ** 2)

                # Classification loss (without temperature scaling)
                classification_logits = self.student_model(student_inputs)
                label_loss = criterion(classification_logits, labels)

                # Total loss
                loss = self.soft_target_loss_weight * distillation_loss + self.ce_loss_weight * label_loss
                running_loss += loss.item()

                # Backpropagation
                loss.backward()
                optimizer.step()

                # Calculate accuracy
                probabilities = torch.sigmoid(classification_logits)
                predicted = (probabilities > 0.5).float()
                n_samples += labels.size(0)
                n_correct += (predicted == labels).sum().item()

            avg_train_loss = running_loss / len(train_loader)
            train_accuracy = 100.0 * n_correct / n_samples

            # Validation step
            val_loss, val_accuracy = self.validate(val_loader, self.student_model, criterion)

            scheduler.step(val_loss)

            # Early stopping and model saving
            if best_val_loss is None or val_loss < best_val_loss - self.delta:
                best_val_loss = val_loss
                counter = 0
                student_best_model_path = f"{self.model_save_directory}/{self.qubit_name}_trace_{self.trace_length}_best_model_kd_student_{self.distillation_setting}_sm.pth"
                torch.save(self.student_model.state_dict(), student_best_model_path)
            else:
                counter += 1
                if counter >= self.patience:
                    print("Early stopping triggered.")
                    break

            # Print statistics
            print(f'Epoch [{epoch + 1}/{self.num_epochs}], '
                  f'Train Loss: {avg_train_loss:.4f}, Train Accuracy: {train_accuracy:.2f}%, '
                  f'Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.2f}%')

        training_duration = time.time() - training_start_time
        print(f'Training completed in {training_duration:.2f} seconds')

        return student_best_model_path

    def test_best_student_model(self, X_test, y_test, best_student_model_path):
        test_loader = get_data_loader(X_test, y_test, batch_size=self.batch_size)

        weights = torch.load(best_student_model_path)
        self.student_model.load_state_dict(weights)
        self.student_model.eval()

        with torch.no_grad():
            n_correct = 0
            n_samples = 0

            for inputs, labels in test_loader:
                inputs = inputs.to(self.device)
                labels = labels.to(self.device).float().unsqueeze(1)

                outputs = self.student_model(inputs)
                probabilities = torch.sigmoid(outputs)
                predicted = (probabilities > 0.5).float()

                n_samples += labels.size(0)
                n_correct += (predicted == labels).sum().item()

            accuracy = 100.0 * n_correct / n_samples
            print(f'Test Best Student Model Accuracy: {accuracy:.2f}%')

        best_model_outcomes = {
            "best_model_path": best_student_model_path,
            "num_non_zero_params": count_nonzero_parameters(self.student_model),
            "num_total_params": count_parameters(self.student_model)
        }
        return accuracy, best_model_outcomes

    def test_best_teacher_model(self, X_test, y_test, best_model_path, trace_length=2*500):
        test_loader = get_data_loader(X_test, y_test, batch_size=self.batch_size)

        weights = torch.load(best_model_path)
        self.teacher_model.load_state_dict(weights)
        self.teacher_model.eval()

        with torch.no_grad():
            n_correct = 0
            n_samples = 0

            for inputs, labels in test_loader:
                inputs = inputs.to(self.device)
                labels = labels.to(self.device).float().unsqueeze(1)

                # Handling ResNet teacher model input
                # if isinstance(self.teacher_model, ResNet):
                #     model_inputs = inputs.view(inputs.size(0), trace_length, 1, 1)  # For teacher model
                # else:
                #     model_inputs = inputs.view(inputs.size(0), trace_length)  # For student model

                model_inputs = inputs.view(inputs.size(0), trace_length)

                outputs = self.teacher_model(model_inputs)
                probabilities = torch.sigmoid(outputs)
                predicted = (probabilities > 0.5).float()

                n_samples += labels.size(0)
                n_correct += (predicted == labels).sum().item()

            accuracy = 100.0 * n_correct / n_samples
            print(f'Test Best Teacher Model Accuracy: {accuracy:.2f}%')

        best_model_outcomes = {
            "best_model_path": best_model_path,
            "num_non_zero_params": count_nonzero_parameters(self.teacher_model),
            "num_total_params": count_parameters(self.teacher_model)
        }
        return accuracy, best_model_outcomes

