
using LinearAlgebra, Optim, Zygote, DataFrames, StatsFuns

"""
    online_estimation(data, knots, initial_alpha, initial_beta, initial_gamma;
                      summary_stats=nothing, 
                      penalty="scad", 
                      xi_1=0.1, xi_2=0.1,
                      max_iter=1000, 
                      tol=1e-5,
                      spl_order=2)

Online updating of the mixture cure model parameters using an MM algorithm.

# Arguments
- `data`: Current batch of data.
- `knots`: Knots for splines.
- `initial_alpha`: Initial values for alpha.
- `initial_beta`: Initial values for beta.
- `initial_gamma`: Initial values for gamma.
- `summary_stats`: A NamedTuple `(N_prev, theta_prev, H_prev)` from the previous batch.
                   If `nothing`, this is treated as the first batch.
- `penalty`: Type of penalty ("scad", "mcp", "lasso", etc. - currently uses SCAD structure).
- `xi_1`: Penalty parameter for alpha.
- `xi_2`: Penalty parameter for beta.
- `spl_order`: Order of the splines (default: 2).

# Returns
- `new_theta`: Updated parameter vector [alpha; beta; gamma].
- `new_stats`: Updated summary statistics for the next batch.
"""
function online_estimation(data, knots, initial_alpha, initial_beta, initial_gamma;
                           summary_stats=nothing, 
                           penalty="scad", 
                           tol=1e-5,
                           spl_order=2)

    p_x = length(initial_alpha)
    q_z = length(initial_beta)
    # J = length(initial_gamma)
    # J = length(knots) + spl_order - 1
    # dim_theta = p_x + q_z + J
    n_k = size(data, 1)
    
    x_names = ["x$(i)" for i in 0:(p_x-1)]
    X_mat = Matrix(data[:, x_names])
    
    z_names = ["z$(i)" for i in 1:q_z]
    Z_mat = Matrix(data[:, z_names])
    
    delta = data.status
    
    I_mat = Ispline(data.tobs, spl_order, knots) # J x n
    M_mat = Mspline(data.tobs, spl_order, knots) # J x n
    
    # If summary_stats is present, we have N_prev, theta_prev, H_prev.
    if !isnothing(summary_stats)
        N_prev = summary_stats.N_prev
        H_prev = summary_stats.H_prev
        theta_prev = vcat(initial_alpha, initial_beta, initial_gamma)
        N_total = N_prev + n_k
    else
        N_total = n_k
    end
    
    xi_1 = 0.05
    xi_2 = 0.05

    # MM Algorithm
    alpha_cur, beta_cur, gamma_cur = initial_alpha, initial_beta, initial_gamma
    for m in 1:1000
        alpha_old = copy(alpha_cur)
        beta_old = copy(beta_cur)
        gamma_old = copy(gamma_cur)
        theta_old_vec = vcat(alpha_old, beta_old, gamma_old)

        # E-Step: Calculate Weights W_i
        # W_i = E[U_i | data, theta^{(m-1)}]
        
        eta_cure = X_mat * alpha_old
        pi_val = 1.0 ./ (1.0 .+ exp.(-eta_cure))
        
        Lambda0 = transpose(I_mat) * gamma_old
        eta_cox = Z_mat * beta_old
        exp_eta = exp.(eta_cox)
        Su = exp.(-Lambda0 .* exp_eta)
        
        denom = max.(1.0 .- pi_val .+ pi_val .* Su, 1e-15) # Numerical stability
        
        # W_i = delta + (1-delta) * (pi * Su) / (1 - pi + pi * Su)
        W = delta .+ (1.0 .- delta) .* (pi_val .* Su) ./ denom
        
        # Surrogate of penalties
        # P'(|beta_old_j|) / (|beta_old_j| + rho)
        rho = 1e-15
        omega_alpha = [scad_deriv(a, xi_1) / (abs(a) + rho) for a in alpha_old]
        omega_beta  = [scad_deriv(b, xi_2) / (abs(b) + rho) for b in beta_old]
        
        # --- M-Step Part 1: Update Alpha ---
        
        function obj_alpha(a)
            term_hist = 0.0 # Historical term
            if !isnothing(summary_stats)
                theta_temp = vcat(a, beta_old, gamma_old)
                diff = theta_temp - theta_prev
                term_hist = 0.5 * dot(diff, H_prev * diff)
            end
            
            eta = X_mat * a
            # log(pi) = -log(1 + exp(-eta)) =  -log1pexp(-eta)
            # log(1-pi) =  -eta + log(pi)
    
            log_pi = -log1pexp.(-eta)
            log_1_minus_pi = -eta .+ log_pi
            
            term_L1 = sum(W .* log_pi .+ (1.0 .- W) .* log_1_minus_pi)
            
            # Penalty part (ignoring constants)
            term_pen = -N_total * sum(0.5 .* omega_alpha .* (a.^2))
            
            return -(term_hist + term_L1 + term_pen) # Return Negative for Minimization
        end

        res_alpha = optimize(obj_alpha, alpha_old, BFGS(), Optim.Options(iterations=50))
        alpha_new = Optim.minimizer(res_alpha)
        
        # --- M-Step Part 2: Update Beta and Gamma ---
    
        function obj_beta_gamma(bg_vec)
            b = bg_vec[1:q_z]
            g = bg_vec[q_z+1 : end]
            
            # Constraint: Gamma must be positive (monotonic cumulative hazard)
            if any(g .<= 0)
                return Inf
            end

            # Historical term
            term_hist = 0.0
            if !isnothing(summary_stats)
                theta_temp = vcat(alpha_new, b, g)
                diff = theta_temp - theta_prev
                term_hist = 0.5 * dot(diff, H_prev * diff)
            end
    
            Lambda0_bg = transpose(I_mat) * g
            lambda0_bg = transpose(M_mat) * g
            
            eta_c = Z_mat * b
            exp_eta_c = exp.(eta_c)
            
            log_Su_bg = -Lambda0_bg .* exp_eta_c # log(Su)
            log_fu_bg = log.(lambda0_bg .+ 1e-15) .+ eta_c .+ log_Su_bg # log(fu)

            term_L2 = sum(W .* (delta .* log_fu_bg .+ (1.0 .- delta) .* log_Su_bg))
            
            # Penalty on Beta
            term_pen = -N_total * sum(0.5 .* omega_beta .* (b.^2))
            
            return -(term_hist + term_L2 + term_pen)
        end

        res_bg = optimize(obj_beta_gamma, vcat(beta_old, gamma_old), BFGS(), Optim.Options(iterations=50))
        bg_new = Optim.minimizer(res_bg)
        
        beta_new = bg_new[1:q_z]
        gamma_new = bg_new[q_z+1 : end]
        
        # --- Check Convergence ---
        theta_new_vec = vcat(alpha_new, beta_new, gamma_new)
        diff_norm = norm(theta_new_vec - theta_old_vec)
        
        alpha_cur = alpha_new
        beta_cur = beta_new
        gamma_cur = gamma_new
        
        if diff_norm < tol
            break
        end
    end
    
    # Calculate new Hessian matrix
    H_current = log_likelihood(alpha_cur, beta_cur, gamma_cur, data, knots; spl_order = spl_order, hess = true)
    
    # Update Summary Statistics
    if !isnothing(summary_stats)
        H_new = H_prev + H_current
    else
        H_new = H_current
    end
    
    new_stats = (N_prev = N_total, H_prev = H_new)
    
    return alpha_cur, beta_cur, gamma_cur, new_stats
end
