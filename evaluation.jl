function c_index(observed_time, delta, risk_score)
    # risk_score: beta * z
    n = length(observed_time)
    numerator = 0.0
    denominator = 0.0

    for i in 1:n
        for j in 1:n
            if i == j 
                continue 
            end
            
            # A pair is "comparable" only if delta=1 and i's time is earlier than j's.

            if (observed_time[i] < observed_time[j]) && (delta[i] == 1)
                denominator += 1.0
                
                # concordance
                if risk_score[i] > risk_score[j]
                    numerator += 1.0
                elseif risk_score[i] == risk_score[j]
                    numerator += 0.5
                end
            end
        end
    end
    
    if denominator == 0
        return NaN
    end

    return numerator / denominator
end

function auc_weighted(incidence_lp, W)
    # incidence_lp: alpha^T * x
    n = length(incidence_lp)

    numerator = 0.0
    denominator = 0.0

    @inbounds for i in 1:n
        for j in 1:n
            if i == j
                continue
            end
    
            weight = W[i] * (1.0 - W[j])
            
            if weight <= 0.0
                continue
            end

            denominator += weight
            
            # Compare predicted value alpha * x
            if incidence_lp[i] > incidence_lp[j]
                numerator += weight
            elseif incidence_lp[i] == incidence_lp[j]
                numerator += 0.5 * weight
            end
        end
    end

    if denominator == 0.0
        return NaN
    end

    return numerator / denominator
end

function model_evaluation(test_data, alpha_hat, beta_hat, gamma_hat, knots, spl_order)
    p_x = length(alpha_hat)
    q_z = length(beta_hat)
    
    x_names = ["x$(i)" for i in 0:(p_x-1)]
    z_names = ["z$(i)" for i in 1:q_z]
    
    X_mat = Matrix(test_data[:, x_names])
    Z_mat = Matrix(test_data[:, z_names])

    observed_time = test_data.tobs
    delta_test = test_data.status

    I_mat = Ispline(observed_time, spl_order, knots) 

    linear_pred = X_mat * alpha_hat
    pi_val = 1.0 ./ (1.0 .+ exp.(-linear_pred))
    
    Lambda0 = transpose(I_mat) * gamma_hat 
    
    risk_score = Z_mat * beta_hat
    exp_risk_score = exp.(risk_score)
    
    Su = exp.(-Lambda0 .* exp_risk_score)
    
    # Calculate Weights W_i
    # W_i = delta + (1-delta) * (pi * Su) / (1 - pi + pi * Su)

    denom = max.(1.0 .- pi_val .+ pi_val .* Su, 1e-15)
    W = delta_test .+ (1.0 .- delta_test) .* (pi_val .* Su) ./ denom

    c_hat = c_index(observed_time, delta_test, risk_score)
    auc_hat = auc_weighted(linear_pred, W)

    return (c_index = c_hat, auc = auc_hat)
end