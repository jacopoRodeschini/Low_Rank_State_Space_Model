# Reproducibility Instrucion for "Multivariate State-Space Model with SPDE Low-Rank Model for High-Dimensional Data"
> Author: Jacopo Rodeschini, Lorenzo Tedesco

This file documents the code associated with the simulation reported in the article and outputs supporting the computational findings and describes how to reproduce simulation results.

## Article Abstract
This paper proposes a novel low-rank approximation to the multivariate State-Space Model for high-dimensional spatio-temporal data, where temporal dynamics follow a first-order autoregressive structure. The Stochastic Partial Differential Equation (SPDE) approach from \cite{lindgren2011explicit} is applied component-wise to the independent-in-time Matérn Gaussian innovation term in the latent equation, assuming component independence. %This results in a sparse representation of the latent process on a finite element mesh, allowing for scalable inference through sparse matrix operations. This results in a low-rank approximation for GPs. It approximates the original GP of mean zero with a low-rank one, which is represented by a linear combination of specified basis functions with random weights. Dependencies among observed components are introduced through a matrix of weights applied to the latent process. %Estimation is achieved using the Kalman filter.  Model parameters are estimated using an Expectation-Maximisation algorithm, which features closed-form updates for most parameters and efficient numerical optimisation for the remaining parameters. We provide theoretical guarantees for the accuracy and convergence of the SPDE-based approximation under fixed-domain asymptotics. Simulation studies and a real-world environmental application illustrate the model’s effectiveness in capturing complex spatio-temporal dependencies while ensuring computational efficiency in high dimensions.


## Contents

* **`main.py`**: Script that runs the simulation to confirm theoretical findings. The simulation results are saved in the `output/` folder.

* **`summarise_results.py`**: Script to organize the simulation results into pandas DataFrames for analysis.

* **`utils/`**: Module containing useful functions.

* **`output/`**: Folder where simulation results are stored.



## Run the simulation and setup Instructions

1. **Download the repository** as a ZIP file named `Low_Rank_State_Space_Model.zip`, and extract it into a folder named `Low_Rank_State_Space_Model`.

2. **Navigate into the folder**:

```bash
cd Low_Rank_State_Space_Model
```

3. **Create the Conda environment** (named `dev`) from the list of required packages:

```bash
conda create --name dev --file environment.txt
```

4. **Activate the environment**:

```bash
conda activate dev
```

5. **Run the `main.py` script**:

```bash
python main.py
```

---

## System Information

The computation time reported in the paper was recorded using Python 3.10.15 on a machine equipped with an **Intel(R) Xeon(R) Platinum 8460Y+** CPU and **1007 GB RAM**.

---


