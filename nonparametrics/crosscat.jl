using Turing, Turing.RandomMeasures
using Plots
using Statistics

# -- generate synthetic data --

# dimensionality
P = 6

# number of samples
N = 20

μ0 = ones(P).^2
y = zeros(Int, P)
for p = 1:P
    K = maximum(y)
    pk = Int[sum(y .== k) for k = 1:K]
    y[p] = rand(ChineseRestaurantProcess(PitmanYorProcess(0.5, 1.0, p-1), pk))
end

K = maximum(y)
μ = Vector{Vector{Vector{Float64}}}(undef, K)
z = Vector{Vector{Int}}(undef, K)
x = Matrix{Float64}(undef, P, N)
for k = 1:K
    μ[k] = Vector{Vector{Float64}}()
    z[k] = zeros(Int, N)
    ps = findall(y .== k)
    for n = 1:N
        J = maximum(z[k])
        nj = Int[sum(z[k] .== j) for j = 1:J]
        z[k][n] = rand(ChineseRestaurantProcess(DirichletProcess(1.0), nj))
        
        if z[k][n] > J
            push!(μ[k], Vector{Float64}(undef, length(ps)))
            for (l, p_) in enumerate(ps)
                μ[k][z[k][n]][l] = rand(Normal(μ0[p_]))
            end
        end
        
        for (l, p_) in enumerate(ps)
            x[p_, n] = rand(Normal(μ[k][z[k][n]][l]))
        end 
    end
end

# -- Turing model for cross categorization --

@model CrossCat(x, μ0) = begin
    
    # dimensionality
    P = size(x, 1)

    # number of samples
    N = size(x, 2)

    # parameters (DP)
    α = 1.0

    # parameters (PYP)
    d = 0.5
    θ = 1.0

    @assert length(μ0) == P

    # assignments to views (dimensions)
    y = tzeros(Int, P)
    for p = 1:P
        K = maximum(y)
        pk = Int[sum(y .== k) for k = 1:K]
        
        # views are assignments are drawn using the CRP of a PYP
        y[p] ~ ChineseRestaurantProcess(PitmanYorProcess(d, θ, p-1), pk)
    end

    # number of active views
    K = maximum(y)
   
    # parameters of univariate distributions (we assume Gaussian for each dimension)
    μ = Vector{Vector{Vector{Float64}}}(undef, K)
    
    # assignments of observations to clusters in each view
    z = TArray{TArray{Int}}(undef, K)
    for k = 1:K
        μ[k] = Vector{Vector{Float64}}()
        z[k] = tzeros(Int, N)

        # find all dimensions in current view 
        ps = findall(y .== k)

        # for each observation
        for n = 1:N
            
            # number of active clusters in the view
            J = maximum(z[k])
            nj = Int[sum(z[k] .== j) for j = 1:J]
            
            # draw assignment to a cluster inside a view from the CRP of a DP
            z[k][n] ~ ChineseRestaurantProcess(DirichletProcess(α), nj)

            # make new cluster if necessary
            if z[k][n] > J
                push!(μ[k], Vector{Float64}(undef, length(ps)))
                for (l, p_) in enumerate(ps)
                    # draw cluster parameters for each dimension in the view 
                    μ[k][z[k][n]][l] ~ Normal(μ0[p_])
                end
            end

            for (l, p_) in enumerate(ps)
                # draw value for each dimension in the view for observation n
                x[p_, n] ~ Normal(μ[k][z[k][n]][l])
            end 
        end
    end
    
    return x
end

# -- sampling --

model = CrossCat(x, μ0);
chain = sample(model, Gibbs(10, HMC(1, 0.1, 5, :μ), PG(10, 1, :z, :y)))
# TODO: this currently doesn't work due to HMC function call problems in Turing#master