using LinearAlgebra, DataFrames

"""
    predict_cure_rate(alpha_val, X)

Predicts the cure rate (probability of being cured, i.e., 1 - π(x)) for given covariates.

# Arguments
- `alpha_val::Vector`: Coefficient vector for the cure (logistic) model.
- `X::Matrix`: Design matrix for cure model covariates (n x p). Should include intercept column if the model was trained with one.

# Returns
- Vector of cure probabilities (length n).
"""
function predict_cure_rate(alpha_val::Vector, X::Matrix)
    # Logistic model: pi(x) = 1 / (1 + exp(-X * alpha))
    eta = X * alpha_val
    pi_val = 1.0 ./ (1.0 .+ exp.(-eta))
    
    # Cure rate is 1 - P(uncured) = 1 - pi
    return 1.0 .- pi_val
end


"""
    predict_survival(alpha_val, beta_val, gamma_val, X, Z, t, knots)

Predicts the overall survival probability S(t | x, z) for individuals.
Assumes 1-to-1 correspondence between rows of X, Z and elements of t.

# Arguments
- `alpha_val::Vector`: Coefficients for cure model.
- `beta_val::Vector`: Coefficients for Cox model.
- `gamma_val::Vector`: Spline coefficients for baseline hazard.
- `X::Matrix`: Covariates for cure model (n x p).
- `Z::Matrix`: Covariates for Cox model (n x q).
- `t::Vector`: Time points to predict survival at (length n).
- `knots::Vector`: Knots used for the I-splines.

# Returns
- Vector of survival probabilities S(t) (length n).
"""
function predict_survival(alpha_val::Vector, beta_val::Vector, gamma_val::Vector, 
                          X::Matrix, Z::Matrix, t::Vector, knots::Vector)
    
    n = size(X, 1)
    if size(Z, 1) != n || length(t) != n
        error("Dimension mismatch: X, Z, and t must have the same number of rows/length.")
    end

    # 1. Uncured Probability: pi(x)
    eta_cure = X * alpha_val
    pi_val = 1.0 ./ (1.0 .+ exp.(-eta_cure))
    
    # 2. Survival of Uncured: Su(t | z) = exp(-Lambda0(t) * exp(beta * z))
    
    # Calculate Baseline Cumulative Hazard Lambda0(t)
    # Ispline returns (J x n) matrix
    I_basis = Ispline(t, 2, knots) 
    
    # Lambda0 = (gamma' * I_basis)' = I_basis' * gamma
    Lambda0 = transpose(I_basis) * gamma_val
    
    # Cox partial likelihood part
    eta_cox = Z * beta_val
    theta = exp.(eta_cox)
    
    # Su(t)
    Su = exp.(-Lambda0 .* theta)
    
    # 3. Overall Survival: S(t) = (1 - pi) + pi * Su
    S = (1.0 .- pi_val) .+ pi_val .* Su
    
    return S
end
