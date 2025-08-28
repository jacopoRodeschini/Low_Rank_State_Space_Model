# Reproducibility Instrucion for "Multivariate Low-Rank State-Space Model with SPDE Approach for High-Dimensional Data"
> Author: Jacopo Rodeschini, Lorenzo Tedesco

This file documents the code associated with the simulation reported in the article and outputs supporting the computational findings and describes how to reproduce simulation results.

## Article Abstract
This paper proposes a novel low-rank approximation to the multivariate State-Space Model. The Stochastic Partial Differential Equation (SPDE) approach is applied component-wise to the independent-in-time Matérn Gaussian innovation term in the latent equation, assuming component independence. This results in a sparse representation of the latent process on a finite element mesh, allowing for scalable inference through sparse matrix operations. Dependencies among observed components are introduced through a matrix of weights applied to the latent process. Model parameters are estimated using the Expectation-Maximisation algorithm, which features closed-form updates for most parameters and efficient numerical routines for the remaining parameters. We prove theoretical results regarding the accuracy and convergence of the SPDE-based approximation under fixed-domain asymptotics. Simulation studies show our theoretical results. We include an empirical application on air quality to demonstrate the practical usefulness of the proposed model, which maintains computational efficiency in high-dimensional settings. In this application, we reduce computation time by about 93\%, with only a 15\% increase in the validation error.


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


