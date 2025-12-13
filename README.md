# Online Mixture Cure Models

This Julia project implements an **Online Mixture Cure Model**. 

## File Structure

- **`gen_mixcure_data.jl`**: Functions to generate simulated data (`simdat_cox_cure`).
- **`online_estimator.jl`**: The online estimation.
- **`likelihood.jl`**: Defines the log-likelihood function.
- **`evaluation.jl`**: Tools for evaluating model performance.
- **`utility_funcs.jl`**: Useful functions.

## Usage

### 1. Setup
Activate the project environment in Julia:

```julia
using Pkg
Pkg.activate(".")
# Pkg.instantiate() # Run this once to install dependencies
```

### 2. Example Workflow

Here is a basic example of how to generate data, initialize the model, and perform online updates:

```julia
using DataFrames, Distributions, LinearAlgebra

include("utility_funcs.jl")
include("gen_mixcure_data.jl")
include("likelihood.jl")
include("online_estimator.jl")
include("evaluation.jl")

n = 2000
alphas_true = [1.0, -1.0, 1.0, 0.0, 0.0] # Cure model coefficients
betas_true  = [1.0, -1.0, 1.0, 0.0, 0.0] # Survival model coefficients
p = length(alphas_true)
q = length(betas_true)
J0 = 5      # Number of spline knots
order = 2   # Spline order

# Generate initial batch of data
data_result = simdat_cox_cure(n, alphas_true, betas_true, baseline="case2", cens_rate_param=0.1, seed=1234)
data = data_result.data

knots = get_knots(data.tobs[data.status .== 1], J0)
num_spline_params = J0 - 2 + order

# Initial parameters
alpha_init = fill(0.0, p)
beta_init  = fill(0.0, q)
gamma_init = fill(0.1, num_spline_params)

# Initial Estimation
alpha_hat, beta_hat, gamma_hat, stats = online_estimation(
    data, knots, alpha_init, beta_init, gamma_init; 
    penalty="none"
)

alpha_hat, beta_hat, gamma_hat, stats = online_estimation(
    data, knots, alpha_hat, beta_hat, gamma_hat; 
    penalty="scad"
)

# Online Update
new_data = simdat_cox_cure(n, alphas_true, betas_true, baseline="case2", cens_rate_param=0.1).data

# Evaluate current model on new data
metrics = model_evaluation(new_data, alpha_hat, beta_hat, gamma_hat, knots, order)
println("Batch 2 - C-index: \$(metrics.c_index), AUC: \$(metrics.auc)")

# Update estimates using summary statistics from previous batch
alpha_hat, beta_hat, gamma_hat, stats = online_estimation(
    new_data, knots, alpha_hat, beta_hat, gamma_hat; 
    summary_stats=stats.summary_stats,
    penalty="scad"
)