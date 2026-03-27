# Online Mixture Cure Models

This repository contains the Julia implementation for OMCM method in "Online Mixture Cure Models for Large-scale Survival Data Streams" by JIN Li, HU Tao, and XIANG Liming.

---


## Repository Structure

```
├── OMCM/
│   ├── Project.toml          # Julia project dependencies
│   ├── Manifest.toml         # Dependency manifest
│   ├── gen_mixcure_data.jl   # Data simulation functions
│   ├── online_estimator.jl   # Core online estimation algorithms
│   ├── likelihood.jl         # Log-likelihood function definitions
│   └── utility_funcs.jl      # Helper functions
└── README.md
```

---

## Acknowledgments

The splines implementation in this repository is adapted from:

> Qiu, M., & Hu, T. (2024). Bayesian transformation model for spatial partly interval-censored data. *Journal of Applied Statistics*, 51(11), 2139-2156.
>

---

## Installation & Setup

1. Clone this repository:
   ```bash
   git clone https://github.com/jinli-stat/online-mixture-cure-models.git
   cd online-mixture-cure-models
   ```

2. Activate and instantiate the Julia environment:
   ```julia
   using Pkg
   Pkg.activate("./OMCM")
   Pkg.instantiate()  # Install dependencies (run once)
   ```

---

## Quick Start

### Basic Usage

```julia
using DataFrames, Distributions, LinearAlgebra

include("OMCM/utility_funcs.jl")
include("OMCM/gen_mixcure_data.jl")
include("OMCM/likelihood.jl")
include("OMCM/online_estimator.jl")

# Simulation parameters
n = 2000
alphas_true = [1.0, -1.0, 1.0, 0.0, 0.0]  # Cure model coefficients
betas_true  = [1.0, -1.0, 1.0, 0.0, 0.0]  # Survival model coefficients
p = length(alphas_true)
q = length(betas_true)
J0 = 5 
order = 3

# Generate initial batch of data
data_result = simdat_cox_cure(n, alphas_true, betas_true; cens_rate_param=0.1, seed=1234)
data = data_result.data

knots = get_knots(data.tobs[data.status .== 1], J0)
num_spline_params = J0 - 2 + order

# Initial parameters
alpha_init = fill(0.0, p)
beta_init  = fill(0.0, q)
gamma_init = fill(0.1, num_spline_params)

# Initial estimation
alpha_hat, beta_hat, gamma_hat, stats = online_estimation(
    data, knots, alpha_init, beta_init, gamma_init;
    penalty="none", spl_order=order
)

alpha_hat, beta_hat, gamma_hat, stats = online_estimation(
    data, knots, alpha_hat, beta_hat, gamma_hat;
    penalty="scad", spl_order=order
)

# Online update with new batch
new_data = simdat_cox_cure(n, alphas_true, betas_true; cens_rate_param=0.1).data

alpha_hat, beta_hat, gamma_hat, stats = online_estimation(
    new_data, knots, alpha_hat, beta_hat, gamma_hat;
    summary_stats=stats,
    penalty="scad", spl_order=order
)
```

---

## Reproducing Paper Results

The following code reproduces the results for Table 2 with K=10 batches and MIC penalty.

**Runtime**: Approximately 7 minutes on the authors' server (may vary based on hardware).

```julia
using ProgressMeter
using PrettyTables

p = 50
n = 500
order = 2
pen = "mic"
alphas_true, betas_true = get_true_params(p)

sim_res = DataFrame(
    SimID=Int[],
    TP_alpha=Float64[], FP_alpha=Float64[], MSE_alpha=Float64[],
    TP_beta=Float64[], FP_beta=Float64[], MSE_beta=Float64[]
)

@showprogress Threads.@threads for sim in 1:100

    seed = sim * 12345 + 5500
    Random.seed!(seed)

    d1_res = simdat_cox_cure(n, alphas_true, betas_true; cens_rate_param=0.1)
    d1 = d1_res.data

    knots = get_knots(d1.tobs[d1.status .== 1], 10)

    a_hat = zeros(p) .+ 0.2
    b_hat = zeros(p) .+ 0.2
    g_hat = fill(0.2, 10 - 2 + order)

    a_hat, b_hat, g_hat, stats = online_estimation(d1, knots, a_hat, b_hat, g_hat; penalty="none", spl_order=2)
    a_hat, b_hat, g_hat, stats = online_estimation(d1, knots, a_hat, b_hat, g_hat; penalty=pen, summary_stats=nothing, spl_order=order)

    for k in 2:10
        d_new_res = simdat_cox_cure(n, alphas_true, betas_true; cens_rate_param=0.1)
        d_new = d_new_res.data

        a_hat, b_hat, g_hat, stats = online_estimation(d_new, knots, a_hat, b_hat, g_hat; summary_stats=stats, penalty=pen, spl_order=order)
    end

    tp_a, fp_a, mse_a = calc_metrics(a_hat, alphas_true)
    tp_b, fp_b, mse_b = calc_metrics(b_hat, betas_true)

    push!(sim_res, (sim, tp_a, fp_a, mse_a, tp_b, fp_b, mse_b))
end

summary_res = combine(sim_res,
    :TP_alpha => mean => :TP_alpha,
    :FP_alpha => mean => :FP_alpha,
    :MSE_alpha => mean => :MSE_alpha,
    :TP_beta  => mean => :TP_beta,
    :FP_beta  => mean => :FP_beta,
    :MSE_beta => mean => :MSE_beta
)

pretty_table(summary_res)
```

**Expected Output:**
```
┌──────────┬──────────┬────────────┬─────────┬─────────┬────────────┐
│ TP_alpha │ FP_alpha │  MSE_alpha │ TP_beta │ FP_beta │   MSE_beta │
│  Float64 │  Float64 │    Float64 │ Float64 │ Float64 │    Float64 │
├──────────┼──────────┼────────────┼─────────┼─────────┼────────────┤
│      3.0 │     0.03 │ 0.00938388 │     3.0 │     0.0 │ 0.00372337 │
└──────────┴──────────┴────────────┴─────────┴─────────┴────────────┘
```

---

## Citation

```bibtex
@unpublished{jin2026submitted,
  title   = {Online mixture cure models for large-scale survival data streams},
  author  = {Jin, Li and Hu, Tao and Xiang, Liming},
  year    = {2026},
  note    = {Submitted for publication},
}
```

---

## License

This project is licensed under the GNU General Public License v3.0 (GPL-3.0).
