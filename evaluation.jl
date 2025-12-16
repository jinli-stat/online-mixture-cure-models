using Statistics
using Survival

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

function auc_td(case_scores::Vector{Float64}, case_weights::Vector{Float64}, control_scores::Vector{Float64}, control_weights::Vector{Float64})
    numerator = 0.0
    denominator = 0.0
    
    n_case = length(case_scores)
    n_control = length(control_scores)
    
    if n_case == 0 || n_control == 0
        return NaN
    end

    for i in 1:n_case
        score_i = case_scores[i]
        w_i = case_weights[i]
        
        for j in 1:n_control
            score_j = control_scores[j]
            w_j = control_weights[j]
            
            weight_pair = w_i * w_j
            denominator += weight_pair
            
            # Case (Uncured) 分数应该比 Control (Cured) 高
            if score_i > score_j
                numerator += weight_pair
            elseif score_i == score_j
                numerator += 0.5 * weight_pair
            end
        end
    end
    
    if denominator == 0.0 return NaN end
    return numerator / denominator
end

function get_km_survival(km::KaplanMeier, t::Float64)
    idx = searchsortedlast(km.events.time, t)
    return idx == 0 ? 1.0 : km.survival[idx]
end

function compute_uno_data(T::Vector{Float64}, delta::Vector{Int}, scores::Vector{Float64}, eval_t::Float64)
    censoring_status = 1 .- delta
    km_censoring = fit(KaplanMeier, T, censoring_status)
    
    G_at_eval_t = max(get_km_survival(km_censoring, eval_t), 1e-5)
    
    case_scores = Float64[]
    case_weights = Float64[]
    control_scores = Float64[]
    control_weights = Float64[]
    
    for i in 1:length(T)
        G_Ti = max(get_km_survival(km_censoring, T[i]), 1e-5)
        
        if T[i] <= eval_t && delta[i] == 1
            # Case Group
            push!(case_scores, scores[i])
            push!(case_weights, 1.0 / G_Ti)
        elseif T[i] > eval_t
            # Control Group
            push!(control_scores, scores[i])
            push!(control_weights, 1.0 / G_at_eval_t)
        end
    end
    
    return case_scores, case_weights, control_scores, control_weights
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

    linear_pred = X_mat * alpha_hat

    risk_score = Z_mat * beta_hat

    c_hat = c_index(observed_time, delta_test, risk_score)

    eval_t = maximum(test_data.tobs[test_data.status .== 1]) 
    
    case_s, case_w, ctrl_s, ctrl_w = compute_uno_data(observed_time, delta_test, linear_pred, eval_t)
    
    td_auc = auc_td(case_s, case_w, ctrl_s, ctrl_w)

    return (c_index = c_hat, auc = td_auc)
end

function calc_metrics(est::Vector{Float64}, true_val::Vector{Float64})
    thresh = 1e-5
    
    est_active = abs.(est) .> thresh
    true_active = abs.(true_val) .> thresh
    
    tp = sum(est_active .& true_active)
    fp = sum(est_active .& (.!true_active))
    
    mse = sum((est .- true_val).^2)
    
    return tp, fp, mse
end
