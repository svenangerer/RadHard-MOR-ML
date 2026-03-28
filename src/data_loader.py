import torch
from torch.utils.data import TensorDataset, DataLoader
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd

def load_and_prep_data(batch_size=1024, test_size=0.2, random_state=42):
    """
    Fetches the hls4ml_lhc_jets_hlf dataset, preprocesses it, 
    and returns PyTorch DataLoaders.
    """
    print("Fetching dataset from OpenML (this might take a minute on first run)...")
    # Fetch the High-Level Features (HLF) jet dataset
    data = fetch_openml('hls4ml_lhc_jets_hlf', version=1, as_frame=True, parser='auto')
    
    X = data.data
    y = data.target
    
    # Convert targets to numeric if they are categorical strings
    if y.dtype == 'O' or isinstance(y.dtype, pd.CategoricalDtype):
        y = y.astype('category').cat.codes

    print(f"Dataset loaded! Shape: {X.shape}")

    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # Scale the features (Crucial for Neural Networks and SVD/MOR)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Convert to PyTorch Tensors
    X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train.values, dtype=torch.long)
    
    X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test.values, dtype=torch.long)

    # Create PyTorch Datasets
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    print("DataLoaders successfully created.")
    return train_loader, test_loader, scaler

if __name__ == "__main__":
    # Quick test to make sure it works when you run this script directly
    train_loader, test_loader, scaler = load_and_prep_data()
    
    # Fetch one batch to verify
    features, labels = next(iter(train_loader))
    print(f"Batch Features shape: {features.shape}")
    print(f"Batch Labels shape: {labels.shape}")