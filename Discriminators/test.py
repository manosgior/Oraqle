from helpers.data_loader import *
from helpers.data_utils import *
from sklearn.model_selection import train_test_split

from networks.Transformer import *

import torch

data_path = "/home/manosgior/Documents/GitHub/KLiNQ/qubit_readout_klinq/data/five_qubit_data"
train_file_name = "DRaw_C_Tr_v0-001"
test_file_name = "DRaw_C_Te_v0-002"
data_config = {
    'train_sample_size': 5,
    'val_sample_size': 5,
    'normalize': 'mean/std',
    'trace_length': 10,
    'val_sampling_mode': 'stratified'  # stratified
}

X_train, y_train = custom_hdf5_data_loader(data_path, train_file_name, "Train", percent=1.0)
X_test, y_test = custom_hdf5_data_loader(data_path, test_file_name, "Test", percent=1.0)

X_full = np.concatenate((X_train, X_test), axis=0)
y_full = np.concatenate((y_train, y_test), axis=0)

#del X_test X_train y_test y_train

X_train, X_test, y_train, y_test = train_test_split(
    X_full, 
    y_full, 
    test_size=0.2, 
    random_state=42, # For reproducibility
    stratify=y_full    # Ensures same class balance in train/test
)

X_train_final, X_val, y_train_final, y_val = train_test_split(
    X_train, 
    y_train, 
    test_size=0.2, 
    random_state=42, # Use the same random state for consistency
    stratify=y_train
)

BATCH_SIZE = 64
train_dataset = QubitTraceDataset(X_train_final, y_train_final)
val_dataset = QubitTraceDataset(X_val, y_val)
test_dataset = QubitTraceDataset(X_test, y_test)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

print(X_train_final.shape)
print(y_train_final.shape)
print(X_test.shape)
print(y_test.shape)
print(X_val.shape)
print(y_val.shape)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = QubitClassifierTransformer(num_classes=32).to(device) # Assuming 2 classes (0 or 1)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

print("starting training...")

NUM_EPOCHS = 20

for epoch in range(NUM_EPOCHS):
    train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
    val_loss, val_accuracy = evaluate(model, val_loader, criterion, device)
    
    print(f"Epoch {epoch+1}/{NUM_EPOCHS} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Accuracy: {val_accuracy:.2f}%")

# After all epochs are done, run a final evaluation on the test set
print("\n--- Final Test Evaluation ---")
test_loss, test_accuracy = evaluate(model, test_loader, criterion, device)
print(f"Test Loss: {test_loss:.4f} | Test Accuracy: {test_accuracy:.2f}%")

# Save your trained model
torch.save(model.state_dict(), "qubit_transformer_model.pth")