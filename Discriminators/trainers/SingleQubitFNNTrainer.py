import time
import torch
import torch.nn as nn
import torch.optim as optim

from helpers.nn_utils import get_data_loader, get_device


class SingleTrainer:
    def __init__(self, model, nn_config, config):
        self.trace_length = nn_config.get('trace_length', None)
        self.qubit_name = nn_config['qubit_name']

        self.num_epochs = nn_config.get('num_epochs', 300)
        self.batch_size = nn_config.get('batch_size', 512)
        self.learning_rate = nn_config.get('learning_rate', 0.001)
        self.device_type = nn_config.get('device', 'mps')
        self.patience = nn_config.get('patience', 10)
        self.delta = nn_config.get('delta', 0)

        self.device = get_device(self.device_type)
        self.model = model.to(self.device)
        self.model_save_directory = config.get('model_storage_path', './')

    def validate(self, loader, model, criterion):
        model.eval()
        total_loss = 0
        n_correct = 0
        n_samples = 0

        with torch.no_grad():
            for qi_values, labels in loader:
                qi_values = qi_values.to(self.device)
                labels = labels.to(self.device).float()
                labels = labels.unsqueeze(1)
                outputs = model(qi_values)
                loss = criterion(outputs, labels)
                total_loss += loss.item()

                probabilities = torch.sigmoid(outputs)  # Convert logits to probabilities
                predicted = (probabilities > 0.5).float()

                n_samples += labels.size(0)
                n_correct += (predicted == labels).sum().item()

        avg_loss = total_loss / len(loader)
        accuracy = 100.0 * n_correct / n_samples

        return avg_loss, accuracy

    def train(self, X_train, y_train, X_val, y_val):

        criterion, optimizer, scheduler = self.loss_optimizer(self.model, self.device, self.learning_rate)

        train_loader = get_data_loader(X_train, y_train, self.batch_size)
        val_loader = get_data_loader(X_val, y_val, self.batch_size)

        avg_train_loss, train_accuracy = None, None
        best_score, best_model_path = None, None
        counter = 0
        training_start_time = time.time()

        n_total_steps = len(train_loader)
        for epoch in range(self.num_epochs):
            self.model.train()  # Set the model to training mode
            running_loss = 0.0
            n_correct = 0
            n_samples = 0
            for i, (qi_values, labels) in enumerate(train_loader):
                qi_values = qi_values.to(self.device)
                labels = labels.to(self.device).float()
                labels = labels.unsqueeze(1)
                # Forward pass
                outputs = self.model(qi_values)
                loss = criterion(outputs, labels)
                running_loss += loss.item()

                probabilities = torch.sigmoid(outputs)  # Convert logits to probabilities
                predicted = (probabilities > 0.5).float()

                n_samples += labels.size(0)
                n_correct += (predicted == labels).sum().item()

                # Backward pass and optimization
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()


            avg_train_loss = running_loss / len(train_loader)
            train_accuracy = 100.0 * n_correct / n_samples

            # Validation step
            val_loss, val_accuracy,  = self.validate(
                val_loader, self.model, criterion)

            scheduler.step(val_loss)

            if best_score is None or val_loss < best_score - self.delta:
                best_score = val_loss
                counter = 0
                best_model_path = f"{self.model_save_directory}/{self.qubit_name}_trace_{self.trace_length}_best_model.pth"
                torch.save(self.model.state_dict(), best_model_path)
            else:
                counter += 1
                if counter >= self.patience:
                    print("Early stopping triggered.")
                    break

            # Print statistics
            print(f'Epoch [{epoch + 1}/{self.num_epochs}], '
                  f'Train Loss: {avg_train_loss:.4f}, Train Accuracy: {train_accuracy:.2f}%, '
                  f'Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.2f}%, ')

        training_duration = time.time() - training_start_time
        print(f'Training completed in {training_duration:.2f} seconds')

        return avg_train_loss, train_accuracy, best_model_path

    def test(self, X_test, y_test):

        test_loader = get_data_loader(X_test, y_test, batch_size=self.batch_size)
        self.model.eval()
        with torch.no_grad():
            n_correct = 0
            n_samples = 0

            for qi_values, labels in test_loader:
                qi_values = qi_values.to(self.device)
                labels = labels.to(self.device).float()
                labels = labels.unsqueeze(1)

                outputs = self.model(qi_values)
                probabilities = torch.sigmoid(outputs)  # Convert logits to probabilities
                predicted = (probabilities > 0.5).float()  # Classify as 0 or 1 based on the threshold

                n_samples += labels.size(0)
                n_correct += (predicted == labels).sum().item()

            accuracy = 100.0 * n_correct / n_samples
            print(f'Test accuracy: {accuracy:.2f}%')
        return accuracy

    def loss_optimizer(self, model, device, lr=0.001):
        loss = nn.BCEWithLogitsLoss().to(device)

        # Initialize the optimizer with the model parameters
        optimizer = optim.Adam(model.parameters(), lr=self.learning_rate)

        # Initialize a learning rate scheduler to adjust the learning rate based on validation loss
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.75,
            patience=self.patience,
            cooldown=2
        )
        return loss, optimizer, scheduler

    def test_best_model(self, X_test, y_test, best_model_path):

        test_loader = get_data_loader(X_test, y_test, batch_size=self.batch_size)

        weights = torch.load(best_model_path)
        self.model.load_state_dict(weights)
        self.model.eval()

        with torch.no_grad():
            n_correct = 0
            n_samples = 0

            for qi_values, labels in test_loader:
                qi_values = qi_values.to(self.device)
                labels = labels.to(self.device)
                labels = labels.unsqueeze(1)

                outputs = self.model(qi_values)
                probabilities = torch.sigmoid(outputs)  # Convert logits to probabilities
                predicted = (probabilities > 0.5).float()  # Classify as 0 or 1 based on the threshold

                n_samples += labels.size(0)
                n_correct += (predicted == labels).sum().item()

            accuracy = 100.0 * n_correct / n_samples
            print(f'Test Best Model Accuracy: {accuracy:.2f}%')
        return accuracy