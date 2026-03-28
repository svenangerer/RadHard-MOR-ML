import torch
import torch.nn as nn

class JetClassifierMLP(nn.Module):
    def __init__(self, input_dim=16, num_classes=5):
        """
        A baseline Multi-Layer Perceptron (MLP) for the CERN Jet dataset.
        We keep the layers explicitly separated (fc1, fc2, etc.) rather than 
        using nn.Sequential so that we can easily access and compress their 
        individual weight matrices later using SVD.
        """
        super(JetClassifierMLP, self).__init__()
        
        # Define the network architecture
        # 64, 32, 32 is a standard baseline for this specific dataset
        self.fc1 = nn.Linear(input_dim, 64)
        self.relu1 = nn.ReLU()
        
        self.fc2 = nn.Linear(64, 32)
        self.relu2 = nn.ReLU()
        
        self.fc3 = nn.Linear(32, 32)
        self.relu3 = nn.ReLU()
        
        # Output layer
        self.fc4 = nn.Linear(32, num_classes)

    def forward(self, x):
        """
        The forward pass of the network.
        """
        x = self.fc1(x)
        x = self.relu1(x)
        
        x = self.fc2(x)
        x = self.relu2(x)
        
        x = self.fc3(x)
        x = self.relu3(x)
        
        x = self.fc4(x)
        # Note: We do not apply Softmax here because PyTorch's 
        # CrossEntropyLoss function applies it automatically during training.
        return x

if __name__ == "__main__":
    # Quick test to instantiate the model and check its structure
    model = JetClassifierMLP()
    print(model)
    
    # Calculate the total number of parameters (weights + biases)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nTotal trainable parameters: {total_params}")
    
    # Test a dummy forward pass
    dummy_input = torch.randn(10, 16) # Batch of 10 samples, 16 features each
    dummy_output = model(dummy_input)
    print(f"Output shape (Batch Size, Num Classes): {dummy_output.shape}")