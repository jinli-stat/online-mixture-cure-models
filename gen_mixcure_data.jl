using Random, Distributions, DataFrames, LinearAlgebra

function simdat_cox_cure(n::Int, 
                         alphas::Vector{Float64}, 
                         betas::Vector{Float64};
                         baseline::String="case1", 
                         cens_rate_param::Float64=0.2, 
                         seed::Union{Int, Nothing}=nothing)
    
    # Set random seed
    if !isnothing(seed)
        Random.seed!(seed)
    end

    # Define the inverse of the Cumulative Baseline Hazard function
    if baseline == "case1"
        inv_Lambda0 = x -> 10.0 * x
    elseif baseline == "case2"
        inv_Lambda0 = x -> sqrt(x / 0.2)
    else
        error("Unsupported baseline case: $baseline")
    end

    # Generate Covariates X for the Cure Model
    # 'alphas' includes the intercept parameter
    p_x = length(alphas) - 1

    # cov_matrix_z[i, j] = 0.5^|i-j|
    cov_matrix_x = [0.5^abs(i - j) for i in 1:(p_x), j in 1:(p_x)]
    mu_vector_x = zeros(p_x)

    mv_dist_x = MvNormal(mu_vector_x, cov_matrix_x)
    X_covs = transpose(rand(mv_dist_x, n))

    X = hcat(ones(n), X_covs)

    # Calculate Uncured Probability
    eta_cure = X * alphas
    p_uncured = 1.0 ./ (1.0 .+ exp.(-eta_cure))

    # Generate Cure Status Indicator: 1 = uncured (susceptible), 0 = cured
    uncured_idx = rand.(Bernoulli.(p_uncured))

    # Generate Covariates Z for the Survival Model
    p_z = length(betas)

    # cov_matrix_z[i, j] = 0.5^|i-j|
    cov_matrix_z = [0.5^abs(i - j) for i in 1:(p_z), j in 1:(p_z)]
    mu_vector_z = zeros(p_z)

    mv_dist_z = MvNormal(mu_vector_z, cov_matrix_z)
    Z = transpose(rand(mv_dist_z, n))

    # Generate Event Times T
    U = rand(Uniform(0, 1), n)
    eta_surv = Z * betas
    base_val = -log.(U) ./ exp.(eta_surv)

    T_event = inv_Lambda0.(base_val)

    # For cured individuals, set time to a very large value (Inf)
    T_event[uncured_idx .== 0] .= Inf

    # Generate Censoring Times C and Calculate Observed Data
    C = rand(Exponential(1.0/cens_rate_param), n)
    
    tobs = min.(T_event, C)
    status = Int.(T_event .<= C)

    # Structure the Output
    simdata = DataFrame()
    simdata.tobs = tobs
    simdata.status = status

    # Add X columns (named x0, x1, ...)
    for i in 0:p_x
        simdata[!, "x$i"] = X[:, i+1]
    end

    # Add Z columns (named z1, z2, ...)
    for i in 1:p_z
        simdata[!, "z$i"] = Z[:, i]
    end

    # Actual proportion of cured subjects in the sample
    cure_rate_sample = 1.0 - mean(uncured_idx)
    # Overall censoring rate in the sample
    censor_rate_sample = 1.0 - mean(status)

    return (
        data = simdata,
        cure_rate = cure_rate_sample,
        censor_rate = censor_rate_sample
    )
end

function get_true_params(p::Int)
    alpha = zeros(p)
    if p >= 3
        alpha[1:3] .= [0.6, 0.8, -0.8]
    else
        alpha[1:p] .= [0.6, 0.8, -0.8][1:p]
    end
    
    beta = zeros(p)
    if p >= 3
        beta[1:3] .= [0.6, 0.8, -0.8]
    else
        beta[1:p] .= [0.6, 0.8, -0.8][1:p]
    end
    
    return alpha, beta
end



# n = 1000
# alphas_input = [1.0, -1.0, 1.0,0,0,0,0,0]
# betas_input = [1.0, -1.0, 1.0,0,0,0,0,0]
# result = simdat_cox_cure(n, alphas_input, betas_input, baseline="case2", cens_rate_param=0.1, seed=123)
# result = simdat_cox_cure(n, alphas_input, betas_input, baseline="case2", cens_rate_param=0.2, seed=123)
# println("Cure Rate: ", result.cure_rate)
# println("Censoring Rate: ", result.censor_rate)
# println(first(result.data, 6))

# function simdat_cox_cure(n::Int, 
#                          alphas::Vector{Float64}, 
#                          betas::Vector{Float64};
#                          baseline::String="case1", 
#                          cens_rate_param::Float64=0.2, 
#                          seed::Union{Int, Nothing}=nothing)
    
#     # Set random seed
#     if !isnothing(seed)
#         Random.seed!(seed)
#     end

#     # Define the inverse of the Cumulative Baseline Hazard function
#     if baseline == "case1"
#         inv_Lambda0 = x -> 10.0 * x
#     elseif baseline == "case2"
#         inv_Lambda0 = x -> sqrt(x / 0.2)
#     else
#         error("Unsupported baseline case: $baseline")
#     end

#     # Generate Covariates X for the Cure Model
#     # 'alphas' includes the intercept parameter
#     p_x = length(alphas) - 1

#     # cov_matrix_z[i, j] = 0.5^|i-j|
#     cov_matrix_x = [0.5^abs(i - j) for i in 1:(p_x-1), j in 1:(p_x-1)]
#     mu_vector_x = zeros(p_x-1)

#     mv_dist_x = MvNormal(mu_vector_x, cov_matrix_x)
#     X_continuous_covs = transpose(rand(mv_dist_x, n))
#     X_discrete_covs = rand(Bernoulli(0.5), n)

#     X = hcat(ones(n), X_discrete_covs, X_continuous_covs)

#     # Calculate Uncured Probability
#     eta_cure = X * alphas
#     p_uncured = 1.0 ./ (1.0 .+ exp.(-eta_cure))

#     # Generate Cure Status Indicator: 1 = uncured (susceptible), 0 = cured
#     uncured_idx = rand.(Bernoulli.(p_uncured))

#     # Generate Covariates Z for the Survival Model
#     p_z = length(betas)

#     # cov_matrix_z[i, j] = 0.5^|i-j|
#     cov_matrix_z = [0.5^abs(i - j) for i in 1:(p_z-1), j in 1:(p_z-1)]
#     mu_vector_z = zeros(p_z-1)

#     mv_dist_z = MvNormal(mu_vector_z, cov_matrix_z)
#     Z_continuous_covs = transpose(rand(mv_dist_z, n))
#     Z_discrete_covs = rand(Bernoulli(0.5), n)
#     Z = hcat(Z_discrete_covs, Z_continuous_covs)

#     # Generate Event Times T
#     U = rand(Uniform(0, 1), n)
#     eta_surv = Z * betas
#     base_val = -log.(U) ./ exp.(eta_surv)

#     T_event = inv_Lambda0.(base_val)

#     # For cured individuals, set time to a very large value (Inf)
#     T_event[uncured_idx .== 0] .= Inf

#     # Generate Censoring Times C and Calculate Observed Data
#     C = rand(Exponential(1.0/cens_rate_param), n)
    
#     tobs = min.(T_event, C)
#     status = Int.(T_event .<= C)

#     # Structure the Output
#     simdata = DataFrame()
#     simdata.tobs = tobs
#     simdata.status = status

#     # Add X columns (named x0, x1, ...)
#     for i in 0:p_x
#         simdata[!, "x$i"] = X[:, i+1]
#     end

#     # Add Z columns (named z1, z2, ...)
#     for i in 1:p_z
#         simdata[!, "z$i"] = Z[:, i]
#     end

#     # Actual proportion of cured subjects in the sample
#     cure_rate_sample = 1.0 - mean(uncured_idx)
#     # Overall censoring rate in the sample
#     censor_rate_sample = 1.0 - mean(status)

#     return (
#         data = simdata,
#         cure_rate = cure_rate_sample,
#         censor_rate = censor_rate_sample
#     )
# end
