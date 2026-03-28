import torch
import torch.nn as nn

def compress_linear_layer(layer, k):
    """
    Takes a standard nn.Linear layer and applies SVD (Model Order Reduction).
    It physically splits the layer into two sequential layers, keeping only 
    the top 'k' singular values/modes.
    
    Args:
        layer (nn.Linear): The original, trained linear layer.
        k (int): The number of singular values (modes) to keep.
        
    Returns:
        nn.Sequential: A module containing the two new, compressed layers.
    """
    # 1. Extract the weight matrix W
    W = layer.weight.data
    
    # 2. Perform SVD
    # W = U * S * V^H
    # U = Output modes, S = Singular values ("dials"), Vh = Input modes
    U, S, Vh = torch.linalg.svd(W, full_matrices=False)
    
    # 3. Truncate (The MOR step: dropping the weak modes)
    U_k = U[:, :k]
    S_k = S[:k]
    Vh_k = Vh[:k, :]
    
    # 4. Create the two new layers
    in_features = layer.in_features
    out_features = layer.out_features
    
    # Layer 1: Projects the input onto the 'k' most important input modes
    # Shape: (k, in_features)
    layer1 = nn.Linear(in_features, k, bias=False)
    layer1.weight.data = Vh_k
    
    # Layer 2: Scales by the "dials" (S_k) and reconstructs the output modes
    # We absorb the singular values (S_k) into U_k for this layer's weights.
    # Shape: (out_features, k)
    layer2 = nn.Linear(k, out_features, bias=True)
    layer2.weight.data = U_k @ torch.diag(S_k)
    
    # Copy over the original bias if it exists
    if layer.bias is not None:
        layer2.bias.data = layer.bias.data
        
    # 5. Package them together
    return nn.Sequential(layer1, layer2)

def compress_model(model, k_dict):
    """
    Iterates through a model and replaces specific layers with compressed versions.
    
    Args:
        model (nn.Module): The full baseline model.
        k_dict (dict): A dictionary mapping layer names to their 'k' value. 
                       e.g., {'fc1': 16, 'fc2': 8}
    """
    # We use a simple approach here since we know our JetClassifierMLP structure
    for name, k in k_dict.items():
        if hasattr(model, name):
            original_layer = getattr(model, name)
            if isinstance(original_layer, nn.Linear):
                print(f"Compressing {name} down to {k} modes...")
                compressed_layer = compress_linear_layer(original_layer, k)
                setattr(model, name, compressed_layer)
    return model

if __name__ == "__main__":
    from model import JetClassifierMLP
    
    # 1. Instantiate the baseline model
    model = JetClassifierMLP()
    print("--- Original Model ---")
    print(model)
    orig_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Original Parameters: {orig_params}")
    
    # 2. Define how many modes to keep for each layer
    # fc1 is (64, 16), fc2 is (32, 64), fc3 is (32, 32). 
    # Let's drastically reduce them.
    k_config = {
        'fc1': 8,   # Keep top 8 modes
        'fc2': 16,  # Keep top 16 modes
        'fc3': 16   # Keep top 16 modes
    }
    
    # 3. Compress!
    compressed_model = compress_model(model, k_config)
    print("\n--- Compressed Model ---")
    print(compressed_model)
    comp_params = sum(p.numel() for p in compressed_model.parameters() if p.requires_grad)
    print(f"Compressed Parameters: {comp_params}")
    
    # Calculate savings
    reduction = (1 - (comp_params / orig_params)) * 100
    print(f"\nTotal physical footprint reduced by {reduction:.2f}%!")
    print("This means the FPGA needs less memory, less power, and has a smaller target for radiation.")