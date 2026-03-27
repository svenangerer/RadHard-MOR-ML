# RadHard-MOR-ML: Robust Neural Network Compression for FPGAs

## Project Overview

This repository explores the intersection of **High-Performance Machine Learning**, **Model Order Reduction (MOR)**, and **hardware reliability**. The goal is to deploy extremely efficient, highly compressed neural networks onto FPGAs operating in radiation-heavy environments (like space or particle accelerators), using **CERN LHC jet physics data** as a baseline.

We use **Singular Value Decomposition (SVD)**—the discrete equivalent of Proper Orthogonal Decomposition (POD) in MOR—to compress the network. We then use **Monte Carlo simulations** to analyze Single Event Upset (SEU) vulnerabilities, proving that mathematically compressed models inherently reduce the physical "surface area" for radiation strikes, allowing for highly targeted fault tolerance.

---

## Core Concepts

### Model Order Reduction (MOR) via SVD
Treating massive neural network weight matrices as high-dimensional systems and projecting them onto a lower-dimensional orthogonal subspace. We only keep the most critical "spatial modes" (singular vectors) and drop the rest.

### Single Event Upsets (SEUs)
Simulating hardware-level bit flips caused by ionizing radiation hitting FPGA memory blocks (BRAM/DSPs) during inference.

### Monte Carlo Sensitivity Analysis
Randomly injecting bit flips across the network over thousands of inference runs to statistically determine which specific weights or layers cause the most catastrophic failures.

### Selective Hardening
Using **Triple Modular Redundancy (TMR)** or specialized data types only on the most critical, mathematically significant weights identified by the MOR and Monte Carlo steps.

---

## Project Roadmap

1. **Baseline Setup**
   Fetch and preprocess the `hls4ml_lhc_jets_hlf` openML dataset. Train a baseline Multi-Layer Perceptron (MLP) or Convolutional Neural Network (CNN).

2. **MOR Compression**
   Implement SVD-based compression on the trained weight matrices. Analyze the trade-off between the number of retained singular values and the model's accuracy.

3. **SEU Simulation Framework**
   Build a custom wrapper that randomly flips bits in the weight matrices (simulating hardware strikes) during PyTorch/TensorFlow inference.

4. **Monte Carlo Analysis**
   Run massive inference batches with random SEU injections. Map out the network's "vulnerability topography" to find the most sensitive nodes.

5. **Radiation-Resistant MOR**
   Combine the concepts. Prove that the MOR-compressed model is more resilient, and apply selective hardening to the highest-value singular components.

6. **Hardware Translation** *(Future Goal)*
   Export the hardened, compressed model using `hls4ml` for actual FPGA deployment.
