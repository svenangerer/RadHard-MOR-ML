import torch
import struct
import random

def flip_float_bit(value, bit_index):
    """
    Simulates a hardware Single Event Upset (SEU) by flipping a specific bit 
    in a 32-bit floating-point number.
    
    Args:
        value (float): The original weight value.
        bit_index (int): Which bit to flip (0 to 31).
    """
    # 1. Pack the Python float into a 32-bit binary string (IEEE 754 format)
    # '!f' means network byte order (big-endian), standard size float (4 bytes)
    packed_float = struct.pack('!f', value)
    
    # 2. Unpack those bytes into a standard 32-bit integer so we can use bitwise math
    # '!I' means big-endian unsigned integer
    int_rep = struct.unpack('!I', packed_float)[0]
    
    # 3. Flip the bit using the XOR operator (^)
    # (1 << bit_index) creates a mask with a 1 exactly at the target index
    int_rep ^= (1 << bit_index)
    
    # 4. Pack the mutated integer back into bytes, then unpack back to a float
    try:
        mutated_packed = struct.pack('!I', int_rep)
        mutated_float = struct.unpack('!f', mutated_packed)[0]
        return mutated_float
    except OverflowError:
        # Extremely rare edge case where the bit flip creates an invalid float
        return float('nan')

def inject_faults(tensor, num_faults=1):
    """
    Randomly injects 'num_faults' bit flips into a PyTorch tensor.
    This modifies the tensor IN PLACE.
    """
    # Flatten the tensor so we can easily index it with a single random integer
    flat_tensor = tensor.view(-1)
    num_elements = flat_tensor.numel()
    
    if num_elements == 0:
        return

    for _ in range(num_faults):
        # Pick a random weight in the tensor
        target_idx = random.randint(0, num_elements - 1)
        
        # Pick a random bit to flip (0 to 31 for float32)
        target_bit = random.randint(0, 31)
        
        # Extract the original value
        original_val = flat_tensor[target_idx].item()
        
        # Flip it
        mutated_val = flip_float_bit(original_val, target_bit)
        
        # Write it back into the tensor
        flat_tensor[target_idx] = mutated_val

def irradiate_model(model, fault_probability=1e-5):
    """
    Iterates through all trainable weights in a model and applies faults 
    based on a simulated radiation environment (probability of strike per parameter).
    """
    with torch.no_grad(): # We don't want PyTorch tracking gradients during fault injection
        for name, param in model.named_parameters():
            if 'weight' in name: # We usually only target weights, not biases, for SEU sims
                # Calculate how many faults should hit this specific layer based on its size
                expected_faults = int(param.numel() * fault_probability)
                
                # Add some randomness so it's not strictly deterministic
                actual_faults = torch.poisson(torch.tensor([float(expected_faults)])).item()
                actual_faults = int(actual_faults)
                
                if actual_faults > 0:
                    inject_faults(param.data, num_faults=actual_faults)
                    
    return model

if __name__ == "__main__":
    # 1. Let's test the math on a single number
    original_weight = 0.15625
    print(f"Original Weight: {original_weight}")
    
    # Let's flip the 30th bit (one of the exponent bits)
    destroyed_weight = flip_float_bit(original_weight, 30)
    print(f"Weight after SEU on bit 30: {destroyed_weight}")
    print("Notice how flipping an exponent bit creates a massive spike! "
          "This will instantly destroy a neural network's activation.\n")
    
    # 2. Let's test it on a dummy PyTorch tensor
    print("--- Simulating Radiation on a Tensor ---")
    dummy_layer = torch.ones(2, 5) * 0.5  # A small 2x5 layer of 0.5s
    print("Original Tensor:")
    print(dummy_layer)
    
    print("\nInjecting 3 random bit flips...")
    inject_faults(dummy_layer, num_faults=3)
    print(dummy_layer)