using Distributions, Random, DataFrames

const MAX_EXP = log(prevfloat(Inf))
const MIN_EXP = -MAX_EXP
const EPS_FLOAT64 = eps(Float64)

function safe_exp(x)
    return exp(clamp(x, MIN_EXP, MAX_EXP))
end

function safe_log(x)
    return log(max(x, EPS_FLOAT64))
end

function safe_value(x)
    return any(isnan, x) ? Inf : x
end

function penalty_scad(b_hat, tuning_param)
    a_val = 3.7
    abs_b_hat = abs(b_hat)
    if abs_b_hat <= tuning_param
        return tuning_param * abs_b_hat
    elseif tuning_param < abs_b_hat <= a_val * tuning_param
        return (2 * a_val * tuning_param * abs_b_hat - b_hat^2 - tuning_param^2) / (2 * (a_val - 1))
    else
        return (tuning_param^2 * (a_val + 1)) / 2
    end
end

function scad_deriv(theta_val, lambda)
    abs_t = abs(theta_val)
    a = 3.7
    if abs_t <= lambda
        return lambda
    elseif abs_t <= a * lambda
        return (a * lambda - abs_t) / (a - 1)
    else
        return 0.0
    end
end

function penalty_mcp(b_hat, tuning_param)
    a = 2.8
    abs_beta = abs(b_hat)
    if abs_beta <= a * tuning_param
        return tuning_param * abs_beta - abs_beta^2 / (2 * a)
    else
        return (tuning_param^2 * a) / 2
    end
end

function penalty_gselo(b_hat, tuning_param)
    return 1 - safe_exp(-tuning_param * (b_hat^2))
end

function penalty_mic(b_hat, tuning_param)
    temp_val = safe_exp(2 * tuning_param * b_hat^2)
    return (temp_val - 1.0) / (temp_val + 1.0)
end

function get_knots(arr, J0)
    filtered_arr = filter(x -> x != 0 && isfinite(x), arr)
    if isempty(filtered_arr)
        return nothing
    end
    
    max_value = quantile(filtered_arr, 0.99) # maximum(filtered_arr)#
    min_value = quantile(filtered_arr, 0.01)# minimum(filtered_arr)

    knots_J0 = range(min_value, max_value, length=J0)
    knots_J0 = round.(knots_J0, digits=2)
    return knots_J0
end