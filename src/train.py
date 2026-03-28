import torch
import torch.nn as nn
import torch.optim as optim
import os
from src.data_loader import load_and_prep_data
from src.model import JetClassifierMLP

def train_model(epochs=30, batch_size=1024, learning_rate=0.001):
    """
    Trains the baseline MLP on the CERN jet dataset and saves the weights.
    """
    # 1. Setup device (use GPU if available, otherwise CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on device: {device}")

    # 2. Load Data
    train_loader, test_loader, _ = load_and_prep_data(batch_size=batch_size)

    # 3. Initialize Model, Loss, and Optimizer
    model = JetClassifierMLP(input_dim=16, num_classes=5).to(device)
    
    # CrossEntropyLoss automatically applies Softmax to the outputs
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    print("\nStarting Training Loop...")
    
    # 4. The Training Loop
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # Backward pass and optimize
            loss.backward()
            optimizer.step()

            # Track statistics
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()

        train_acc = 100 * correct_train / total_train

        # 5. Validation Phase (Test Set)
        model.eval()
        correct_test = 0
        total_test = 0
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total_test += labels.size(0)
                correct_test += (predicted == labels).sum().item()
                
        test_acc = 100 * correct_test / total_test

        # Print progress every 5 epochs
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"Epoch [{epoch+1}/{epochs}] | Loss: {running_loss/len(train_loader):.4f} | Train Acc: {train_acc:.2f}% | Test Acc: {test_acc:.2f}%")

    # 6. Save the trained model weights
    save_dir = "models/baseline"
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, "baseline_mlp.pth")
    
    torch.save(model.state_dict(), save_path)
    print(f"\nTraining complete. Baseline weights saved to: {save_path}")
    
    return model

if __name__ == "__main__":
    # Run the training loop
    trained_model = train_model(epochs=30)