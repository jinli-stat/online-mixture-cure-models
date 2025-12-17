using LinearAlgebra, Optim, Zygote, DataFrames, StatsFuns, LineSearches

struct estimate_result
    xi_1::Float64
    xi_2::Float64
    alpha::Vector{Float64}
    beta::Vector{Float64}
    gamma::Vector{Float64}
    criterion::Float64
end

function online_estimation(data, knots, initial_alpha, initial_beta, initial_gamma;
                           summary_stats=nothing, 
                           penalty="none", 
                           spl_order=2)

    p_x = length(initial_alpha)
    q_z = length(initial_beta)
    j_g = length(initial_gamma)
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

    function BayIC(alpha_val, beta_val, gamma_val, n)
        lkhd = log_likelihood(alpha_val, beta_val, gamma_val, data, knots; spl_order = spl_order, hess = false)

        DF = count(!iszero, alpha_val) + count(!iszero, beta_val)
        return -2 * lkhd + DF * log(n)
    end

    function evaluate_tuning_param(xi_1, xi_2)
        # MM Algorithm
        alpha_cur, beta_cur, gamma_cur = initial_alpha, initial_beta, initial_gamma
        for m in 1:100
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
            if penalty=="scad"
                omega_alpha = [scad_deriv(a, xi_1) / (abs(a) + rho) for a in alpha_old]
                omega_beta  = [scad_deriv(b, xi_2) / (abs(b) + rho) for b in beta_old]
            end
            
            # --- M-Step Part 1: Update Alpha ---
            
            function obj_alpha(a)
                if penalty == "mic"
                    pen_part = sum(penalty_mic(a, xi_1))
                    a = a.* penalty_mic(a, xi_1)
                end

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

                term_pen = 0.0 
                if penalty == "scad"
                    term_pen = - N_total * sum(0.5 .* omega_alpha .* (a.^2))
                elseif penalty == "mic"
                    term_pen = -log(N_total) * pen_part
                elseif penalty == "none" && p_x >= 10
                    # L2 penalty for high-dim cases to ensure numerical stability
                    term_pen = - N_total * 0.01 * sum(a.^2)
                end

                return -(term_hist + term_L1 + term_pen)
            end

            res_alpha = optimize(obj_alpha, alpha_old, LBFGS(linesearch = LineSearches.BackTracking()), Optim.Options(iterations=20); autodiff = :forward)
            alpha_new = Optim.minimizer(res_alpha)

            if penalty == "mic"
                alpha_new = alpha_new.* penalty_mic(alpha_new, xi_1)
            end
            
            # --- M-Step Part 2: Update Beta and Gamma ---
        
            function obj_beta_gamma(bg_vec)
                b = bg_vec[1:q_z]
                g = bg_vec[q_z+1 : end]

                if penalty == "mic"
                    pen_part = sum(penalty_mic(b, xi_2))
                    b = b.* penalty_mic(b, xi_2)
                end
                
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
                term_pen = 0.0 
                if penalty == "scad"
                    term_pen = -N_total * sum(0.5 .* omega_beta .* (b.^2))
                elseif penalty == "mic"
                    term_pen = -log(N_total) * pen_part
                elseif penalty == "none" && p_x >= 20
                    # L2 penalty for high-dim cases to ensure numerical stability
                    term_pen = - N_total * 0.01 * sum(b.^2)
                end

                return -(term_hist + term_L2 + term_pen)
            end
            res_bg = optimize(obj_beta_gamma, vcat(beta_old, gamma_old), LBFGS(linesearch = LineSearches.BackTracking()), Optim.Options(iterations=20); autodiff = :forward)
            bg_new = Optim.minimizer(res_bg)
            beta_new = bg_new[1:q_z]
            if penalty == "mic"
                beta_new = beta_new.* penalty_mic(beta_new, xi_2)
            end
            gamma_new = bg_new[q_z+1 : end]
            
            # --- Check Convergence ---
            theta_new_vec = vcat(alpha_new, beta_new, gamma_new)

            is_explosion = any(eachindex(theta_new_vec)) do i
                diff = abs(theta_new_vec[i] - theta_old_vec[i])
                return isnan(diff) || diff > 2.0
            end
            if is_explosion
                break
            end
            
            alpha_cur = alpha_new
            beta_cur = beta_new
            gamma_cur = gamma_new

            is_converged = !any(i -> abs(theta_new_vec[i] - theta_old_vec[i]) > 1e-5, eachindex(theta_new_vec))

            if is_converged
                break
            end

        end

        alpha_cur[abs.(alpha_cur) .< 1e-3] .= 0
        beta_cur[abs.(beta_cur) .< 1e-3] .= 0
        BIC_val = BayIC(alpha_cur, beta_cur, gamma_cur, n_k)
        return estimate_result(xi_1, xi_2, alpha_cur, beta_cur, gamma_cur, BIC_val)
    end

    if penalty == "none"
        best_result = evaluate_tuning_param(1.0, 1.0)
    elseif penalty == "mic"
        best_result = evaluate_tuning_param(10.0, 10.0)
    elseif penalty == "scad"
        param1 = exp10.(range(log10(0.001), log10(1.5 * sqrt(log(p_x)/N_total) * (N_total / n_k)^0.35), length=5))
        param1 = round.(param1, digits=4)
        param2 = exp10.(range(log10(0.001), log10(1.5 * sqrt(log(q_z)/N_total) * (N_total / n_k)^0.35), length=5))
        param2 = round.(param2, digits=4) 
        # param1 = [0.001, 0.005, 0.01, 0.05, 0.1]
        # param2 = [0.001, 0.005, 0.01, 0.05, 0.1]
        param_grid = collect(Base.Iterators.product(param1, param2)) |> vec
        tuning_results = [evaluate_tuning_param(xi_1, xi_2) for (xi_1, xi_2) in param_grid]
        if isempty(tuning_results)
            error("No valid tuning parameters.")
        end

        best_idx = argmin(map(r -> r.criterion, tuning_results))
        best_result = tuning_results[best_idx]
    end
    best_alpha = best_result.alpha
    best_beta = best_result.beta
    best_gamma = best_result.gamma
    # println(best_result.xi_1, ", ", best_result.xi_2, " => BIC: ", best_result.criterion)

    # Calculate new Hessian matrix
    H_current = log_likelihood(best_alpha, best_beta, best_gamma, data, knots; spl_order = spl_order, hess = true)
    
    # Update Summary Statistics
    if !isnothing(summary_stats)
        H_new = H_prev + H_current
    else
        H_new = H_current
    end
    
    new_stats = (N_prev = N_total, H_prev = H_new)
    
    return best_alpha, best_beta, best_gamma, new_stats
end
