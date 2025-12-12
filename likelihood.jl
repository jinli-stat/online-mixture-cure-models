using DataFrames, LinearAlgebra, Zygote

function log_likelihood(alpha_val, beta_val, gamma_val, data, knots; spl_order = 2, hess = false)
    
    # Use Zygote.@ignore to treat data extraction and splines as constants
    X, Z, I_mat, M_mat, idx_event, idx_cens = Zygote.@ignore begin
        p = length(alpha_val)
        q = length(beta_val)
        
        # Covariates
        x_names = ["x$(i)" for i in 0:(p-1)]
        X_full = Matrix(data[:, x_names])
        
        z_names = ["z$(i)" for i in 1:q]
        Z_full = Matrix(data[:, z_names])
        
        # Indices for events and censoring
        delta = data.status
        idx_e = findall(x -> x == 1, delta)
        idx_c = findall(x -> x == 0, delta)
        
        # I_spline for Cumulative Hazard, M_spline for Hazard
        I_full = Ispline(data.tobs, spl_order, knots)
        M_full = Mspline(data.tobs, spl_order, knots)
        
        (X_full, Z_full, I_full, M_full, idx_e, idx_c)
    end
    
    # Define the Likelihood Calculation
    function compute_ll(a, b, g)  

        eta_cure = X * a
        pi_val = 1.0 ./ (1.0 .+ exp.(-eta_cure))
        eta_cox = Z * b
        exp_eta = exp.(eta_cox)
        
        # Cumulative Baseline Hazard (Lambda0)
        Lambda0 = transpose(I_mat) * g
        
        # Survival Function for Uncured: S_u = exp(-Lambda0 * exp_eta)
        Su = exp.(-Lambda0 .* exp_eta)
        
        # Add small epsilon for log stability
        eps_val = 1e-15
        
        # --- Part A: Events (delta = 1) ---
        # contribution: log( pi * f_u ) 
        # f_u = lambda0 * exp_eta * Su
        # then: log(pi) + log(lambda0) + eta_cox + log(Su)
        
        pi_e = pi_val[idx_event]
        eta_cox_e = eta_cox[idx_event]
        Su_e = Su[idx_event]
        lambda0_e = transpose(M_mat[:, idx_event]) * g

        ll_events = sum(log.(pi_e) .+ log.(lambda0_e .+ eps_val) .+ eta_cox_e .+ log.(Su_e .+ eps_val))
        
        # --- Part B: Censored (delta = 0) ---
        # contribution: log( 1 - pi + pi * Su )
        
        # Slice relevant vectors for censored
        pi_c = pi_val[idx_cens]
        Su_c = Su[idx_cens]
        
        ll_censored = sum(log.(1.0 .- pi_c .+ pi_c .* Su_c .+ eps_val))
        
        return ll_events + ll_censored
    end
    
    # Calculate Value and Optional Hessian
    if !hess
        return compute_ll(alpha_val, beta_val, gamma_val)
    else
        # Zygote.hessian requires a single vector input function
        params_vec = vcat(alpha_val, beta_val, gamma_val)
        p_len = length(alpha_val)
        q_len = length(beta_val)
        
        function wrapper(values)
            a_sub = values[1:p_len]
            b_sub = values[p_len+1 : p_len+q_len]
            g_sub = values[p_len+q_len+1 : end]
            return compute_ll(a_sub, b_sub, g_sub)
        end
        H_mat = Zygote.hessian(wrapper, params_vec)

        return H_mat
    end
end

