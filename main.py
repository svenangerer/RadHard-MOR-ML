import os
import torch
import pandas as pd

# Import our custom modules from the src/ directory
from src.data_loader import load_and_prep_data
from src.train import train_model
from src.compress import compress_model
from src.monte_carlo import run_monte_carlo
from src.model import JetClassifierMLP

def run_experiment():
    print("==================================================")
    print("  RadHard-MOR-ML: Automated Experiment Pipeline   ")
    print("==================================================\n")
    
    # --- Configuration ---
    EPOCHS = 30
    BATCH_SIZE = 1024
    NUM_SIMULATIONS = 500       # How many Monte Carlo runs per model
    FAULT_PROBABILITY = 1e-4    # Simulated radiation intensity
    
    # Define our compression targets (how many modes to keep per layer)
    # You will tweak these during your actual research to find the sweet spot
    K_CONFIG = {
        'fc1': 16,  # Original is 64x16
        'fc2': 16,  # Original is 32x64
        'fc3': 16   # Original is 32x32
    }
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Hardware backend: {device}\n")

    # --- Phase 1: Data Setup ---
    print("--- Phase 1: Loading Data ---")
    train_loader, test_loader, scaler = load_and_prep_data(batch_size=BATCH_SIZE)
    
    # --- Phase 2: Train Baseline ---
    print("\n--- Phase 2: Training Baseline Model ---")
    baseline_weights_path = "models/baseline/baseline_mlp.pth"
    
    if os.path.exists(baseline_weights_path):
        print(f"Found existing baseline weights at {baseline_weights_path}. Skipping training.")
        baseline_model = JetClassifierMLP().to(device)
        baseline_model.load_state_dict(torch.load(baseline_weights_path, map_location=device))
    else:
        print("No baseline weights found. Training from scratch...")
        baseline_model = train_model(epochs=EPOCHS, batch_size=BATCH_SIZE)
        
    orig_params = sum(p.numel() for p in baseline_model.parameters() if p.requires_grad)
    print(f"Baseline Physical Footprint: {orig_params} trainable parameters.")

    # --- Phase 3: Irradiate Baseline ---
    print("\n--- Phase 3: Monte Carlo SEU Simulation (Baseline) ---")
    baseline_clean_acc, df_baseline = run_monte_carlo(
        baseline_model, test_loader, num_simulations=NUM_SIMULATIONS, fault_prob=FAULT_PROBABILITY
    )
    df_baseline.to_csv("results/logs/baseline_mc_results.csv", index=False)

    # --- Phase 4: MOR Compression ---
    print("\n--- Phase 4: Compressing Model via SVD (MOR) ---")
    # We must instantiate a fresh model and load the weights so we don't accidentally
    # compress a model that has residual radiation damage in its memory
    model_to_compress = JetClassifierMLP().to(device)
    model_to_compress.load_state_dict(torch.load(baseline_weights_path, map_location=device))
    
    compressed_model = compress_model(model_to_compress, K_CONFIG)
    comp_params = sum(p.numel() for p in compressed_model.parameters() if p.requires_grad)
    
    print(f"Compressed Physical Footprint: {comp_params} trainable parameters.")
    print(f"Footprint Reduction: {(1 - comp_params/orig_params)*100:.2f}%")
    
    # Save the compressed weights
    os.makedirs("models/compressed", exist_ok=True)
    torch.save(compressed_model.state_dict(), "models/compressed/compressed_mlp.pth")

    # --- Phase 5: Irradiate Compressed Model ---
    print("\n--- Phase 5: Monte Carlo SEU Simulation (Compressed) ---")
    compressed_clean_acc, df_compressed = run_monte_carlo(
        compressed_model, test_loader, num_simulations=NUM_SIMULATIONS, fault_prob=FAULT_PROBABILITY
    )
    df_compressed.to_csv("results/logs/compressed_mc_results.csv", index=False)

    # --- Phase 6: Quick Summary ---
    print("\n==================================================")
    print("                 EXPERIMENT COMPLETE              ")
    print("==================================================")
    print(f"Baseline Pristine Acc:   {baseline_clean_acc:.2f}%")
    print(f"Compressed Pristine Acc: {compressed_clean_acc:.2f}%")
    print(f"Acc Drop from MOR:       {baseline_clean_acc - compressed_clean_acc:.2f}%")
    print("--------------------------------------------------")
    print(f"Baseline Median under Radiation:   {df_baseline['accuracy'].median():.2f}%")
    print(f"Compressed Median under Radiation: {df_compressed['accuracy'].median():.2f}%")
    print("==================================================")
    print("Check Jupyter notebooks to plot the comparative histograms!")

if __name__ == "__main__":
    # Ensure our output directories exist
    os.makedirs("models/baseline", exist_ok=True)
    os.makedirs("models/compressed", exist_ok=True)
    os.makedirs("results/logs", exist_ok=True)
    os.makedirs("results/plots", exist_ok=True)
    
    run_experiment()