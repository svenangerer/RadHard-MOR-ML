import torch
import copy
import numpy as np
from tqdm import tqdm
import pandas as pd
import os

from data_loader import load_and_prep_data
from model import JetClassifierMLP
from fault_inject import irradiate_model

def evaluate_model(model, test_loader, device):
    """
    Quickly evaluates a model's accuracy on the test set.
    """
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
    return 100 * correct / total

def run_monte_carlo(clean_model, test_loader, num_simulations=1000, fault_prob=1e-4):
    """
    Runs the Monte Carlo SEU simulation pipeline.
    
    Args:
        clean_model: The trained, pristine PyTorch model.
        test_loader: The data loader for the test set.
        num_simulations: How many times to irradiate and test the model.
        fault_prob: The probability of a bit flip per parameter.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    clean_model.to(device)
    
    # 1. Get the baseline (perfect) accuracy
    print("Evaluating pristine model baseline...")
    baseline_acc = evaluate_model(clean_model, test_loader, device)
    print(f"Pristine Baseline Accuracy: {baseline_acc:.2f}%\n")
    
    results = []
    
    print(f"Starting Monte Carlo Simulation ({num_simulations} runs)...")
    print(f"Simulated SEU probability per weight: {fault_prob}")
    
    # 2. The Monte Carlo Loop
    for i in tqdm(range(num_simulations)):
        # CRITICAL: We must create a fresh, uncorrupted copy of the model 
        # for every single simulation run.
        test_subject = copy.deepcopy(clean_model)
        
        # 3. Inject random radiation faults
        test_subject = irradiate_model(test_subject, fault_probability=fault_prob)
        
        # 4. Evaluate the broken model
        degraded_acc = evaluate_model(test_subject, test_loader, device)
        
        # 5. Log the results
        results.append({
            'run_id': i,
            'accuracy': degraded_acc,
            'accuracy_drop': baseline_acc - degraded_acc
        })
        
    return baseline_acc, pd.DataFrame(results)

if __name__ == "__main__":
    # 1. Load the data
    _, test_loader, _ = load_and_prep_data(batch_size=2048)
    
    # 2. Load the trained pristine weights
    # (Assuming you already ran train.py)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = JetClassifierMLP().to(device)
    
    weights_path = "../models/baseline/baseline_mlp.pth"
    try:
        model.load_state_dict(torch.load(weights_path, map_location=device))
        print("Successfully loaded baseline weights.")
    except FileNotFoundError:
        print("ERROR: Baseline weights not found. Please run train.py first!")
        exit()
        
    # 3. Run a small Monte Carlo test (100 runs) to verify the pipeline
    # We use a relatively high fault probability (1e-3) here just to guarantee 
    # we see some massive failures quickly.
    baseline_acc, df_results = run_monte_carlo(
        model, test_loader, num_simulations=100, fault_prob=1e-3
    )
    
    # 4. Output Summary Statistics
    print("\n--- Monte Carlo Results Summary ---")
    print(f"Mean Accuracy: {df_results['accuracy'].mean():.2f}%")
    print(f"Median Accuracy: {df_results['accuracy'].median():.2f}%")
    print(f"Worst Case Scenario: {df_results['accuracy'].min():.2f}%")
    print(f"Standard Deviation: {df_results['accuracy'].std():.2f}%")
    
    # Save the logs to a CSV for later visualization in a Jupyter Notebook
    os.makedirs("../results/logs", exist_ok=True)
    df_results.to_csv("../results/logs/baseline_mc_results.csv", index=False)
    print("\nDetailed results saved to results/logs/baseline_mc_results.csv")