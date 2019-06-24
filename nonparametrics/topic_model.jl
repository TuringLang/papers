using Turing, Turing.RandomMeasures
using Plots, StatsPlots
using Statistics, Random, LinearAlgebra

# -- LDA model with finite number of topics --

@model LDA(w) = begin
    
    # number of topics
    K = 4

    # number of words
    D = 20
    
    # number of documents
    M = size(w,1)
    
    # number of words per document
    N = size(w,2)
    
    # topic distributions
    θ = Vector{Vector}(undef, M)
    for m = 1:M
        θ[m] ~ Dirichlet(K, 1.0)
    end

    # word distributions
    ψ = Vector{Vector}(undef, K)
    for k = 1:K
        ψ[k] ~ Dirichlet(D, 0.01)
    end

    z = Vector{Vector{Int}}(undef, M)

    for m = 1:M
        z[m] = zeros(Int, N)
        for n = 1:N
            # select topic for word n in document m
            z[m][n] ~ Categorical(θ[m])
            
            # select symbol for word n in document m from topic z[m][n]
            w[m,n] ~ Categorical(ψ[z[m][n]])
        end
    end
    return w
end

# number of documents
M = 2

# number of words per document
N = 10

# draw from prior
prior_LDA = LDA(fill(missing, M, N))
X = prior_LDA()

# -- LDA model with infinite many topics (nonparametric) --

@model CRF_LDA(w) = begin

    # number of words
    D = 20
    
    # number of documents
    M = size(w,1)
    
    # number of words per document
    N = size(w,2)
    
    # random measure
    rpm = DirichletProcess(1.0)

    t = zeros(Int, M,N)
    k = Vector{Vector{Int}}(undef, M)

    # foreach document (restaurant)
    for m = 1:M
        
        # foreach word (customer)
        for n = 1:N
            nk = Int[sum(t[m,:] .== j) for j in 1:maximum(t[m,:])]

            # select table t[m,n] for customer n in restaurant m
            t[m,n] ~ ChineseRestaurantProcess(rpm, nk)
        end

        # number of tables at restaurant m
        J = maximum(t[m,:])
        k[m] = zeros(Int, J)

        # foreach table in restaurant m
        for j in 1:J
            kk = m > 1 ? vcat(reduce(vcat, k[1:m-1]), k[m][1:j-1]) : k[m][1:j-1]
            mj = isempty(kk) ? zeros(Int,0) : Int[sum(kk .== d) for d in 1:maximum(kk)]

            # select topic (dish) at table k
            k[m][j] ~ ChineseRestaurantProcess(rpm, mj)
        end
    end
    
    # total number of topics
    K = maximum(map(maximum, k))

    # foreach latent topic
    for k in 1:K
        # draw distribution over symbols
        θ[k] = rand(Dirichlet(D, 0.01))
    end

    # foreach document
    for m = 1:M
        # foreach word
        for n = 1:N
            # select symbol for word n in document m at table t[m,n] with topic (dish) k[m][ t[m,n] ]
            w[m,n] ~ Categorical(θ[k[m][t[m,n]]])
        end
    end
    
    return w
end

# number of documents
M = 2

# number of words per document
N = 10

# draw from the prior
prior_CRF_LDA = CRF_LDA(fill(missing, M, N))
X = prior_CRF_LDA()