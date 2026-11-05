import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader, TensorDataset, RandomSampler, SequentialSampler


def get_data_loader(X, y, batch_size=1024, shuffle=True):
    """
    Prepare DataLoader from the provided feature matrix X and label vector y.

    Args:
        X (np.ndarray or torch.Tensor): The input features for the model.
        y (np.ndarray or torch.Tensor): The corresponding labels for the input features.
        batch_size (int): The size of each batch during training.
        shuffle (bool): Whether to shuffle the data during loading or not.

    Returns:
        DataLoader: The DataLoader object to be used for model training.
    """
    # Check if X and y are numpy arrays and convert them to tensors
    if not isinstance(X, torch.Tensor):
        X = torch.tensor(X, dtype=torch.float32)  # Ensure floating point for features
    if not isinstance(y, torch.Tensor):
        y = torch.tensor(y, dtype=torch.long)  # Ensure long int for categorical labels

    # Assert that the first dimension matches
    assert X.shape[0] == y.shape[0], "The number of samples in X and y must be the same"

    # Create a TensorDataset
    dataset = TensorDataset(X, y)

    # Choose the sampler based on the shuffle parameter
    if shuffle:
        data_sampler = RandomSampler(dataset)
    else:
        data_sampler = SequentialSampler(dataset)

    # Create and return the DataLoader
    return DataLoader(dataset, batch_size=batch_size, sampler=data_sampler, num_workers=5)


def loss_optimizer(model, device, learning_rate=0.001):
    """
    Initializes the loss function, optimizer, and learning rate scheduler for a given model.

    Args:
        model (torch.nn.Module): The model for which to configure the loss and optimizer.
        device (torch.device): The device to which the model and loss function are to be moved.
        learning_rate (float, optional): Initial learning rate for the optimizer.

    Returns:
        tuple: Tuple containing:
            - nn.Module: Loss function.
            - optim.Optimizer: Optimizer configured for the model.
            - optim.lr_scheduler: Learning rate scheduler.
    """
    # Initialize the loss function and move it to the specified device
    loss = nn.CrossEntropyLoss().to(device)

    # Initialize the optimizer with the model parameters
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Initialize a learning rate scheduler to adjust the learning rate based on validation loss
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.75,
        patience=15,
        cooldown=5
        # patience=5,
        # cooldown=2
    )

    return loss, optimizer, scheduler


def get_device(device_type: str = 'cpu') -> torch.device:
    """
    Retrieves a PyTorch device object based on the specified type.

    Args:
        device_type (str, optional): The type of device to use. Defaults to 'cpu'.
            It can be 'cpu' or 'cuda' or other valid PyTorch device strings.

    Returns:
        torch.device: A PyTorch device corresponding to the given type.
    """
    return torch.device(device_type)
