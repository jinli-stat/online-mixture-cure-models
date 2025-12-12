using Random, Distributions, DataFrames, LinearAlgebra

"""
    simdat_cox_cure(n, alphas, betas; baseline="case1", cens_rate_param=0.05, seed=nothing)

Simulate survival data under mixture Cox models.

# Arguments
- `n::Int`: Sample size.
- `alphas::Vector{Float64}`: Coefficients for the cure (logistic) model, including intercept.
- `betas::Vector{Float64}`: Coefficients for the Cox model.
- `baseline::String`: Baseline hazard for the Cox model ("case1" or "case2").
- `cens_rate_param::Float64`: Parameter for censoring rate (rate parameter for Exponential distribution).
- `seed::Union{Int, Nothing}`: Random seed for reproducibility.

# Returns
- A NamedTuple `(data=simdata, cure_rate=cure_rate_sample, censor_rate=censor_rate_sample)`
"""
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

    # Generate normal random variables for predictors
    X_covs = rand(Normal(0, 1), n, p_x)
    # Add intercept column
    X = hcat(ones(n), X_covs)

    # Calculate Uncured Probability
    eta_cure = X * alphas
    p_uncured = 1.0 ./ (1.0 .+ exp.(-eta_cure))

    # Generate Cure Status Indicator: 1 = uncured (susceptible), 0 = cured
    uncured_idx = rand.(Bernoulli.(p_uncured))

    # Generate Covariates Z for the Survival Model
    p_z = length(betas)

    # cov_matrix[i, j] = 0.2^|i-j|
    cov_matrix = [0.2^abs(i - j) for i in 1:p_z, j in 1:p_z]
    mu_vector = zeros(p_z)

    mv_dist = MvNormal(mu_vector, cov_matrix)
    Z = transpose(rand(mv_dist, n))

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

# Sample usage
# n = 1000
# alphas_input = [1.0, -1.0, 1.0]
# betas_input = [1.0, -1.0, 1.0]
# result = simdat_cox_cure(n, alphas_input, betas_input, baseline="case1", cens_rate_param=0.05, seed=123)
# println("Cure Rate: ", result.cure_rate)
# println("Censoring Rate: ", result.censor_rate)
# println(first(result.data, 6))
