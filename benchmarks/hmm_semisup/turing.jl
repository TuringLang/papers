using Random: seed!
seed!(1)

include("data.jl")

data = get_data()

using Turing

using StatsFuns: logsumexp

# FIXME: tag1 + tag2 + tag3 lead to AD error
@model hmm_semisup(K, V, T, T_unsup, w, z, u, alpha, beta, ::Type{Tv}=Vector{Float64}) where {Tv} = begin
    theta = Vector{Tv}(undef, K)
    for k = 1:K
        theta[k] ~ Dirichlet(alpha)
    end
    # theta ~ Multi(Dirichlet(alpha), K)  # tag1
    phi = Vector{Tv}(undef, K)
    for k = 1:K
        phi[k] ~ Dirichlet(beta)
    end
  
    w ~ ArrayDist(Categorical.(phi[z]))
    z[2:end] ~ ArrayDist(Categorical.(theta[z[1:end-1]]))
    # z[2:end] ~ ArrayDist(Categorical.(theta[:,zi] for zi in z[1:end-1]))    # tag2

    # Forward algorithm
    acc, gamma, gamma′ = Tv(undef, K), Tv(undef, K), Tv(undef, K)
    for k in 1:K
        gamma[k] = log(phi[k][u[1]])
    end
    for t in 2:T_unsup
        for k in 1:K
            for j in 1:K
              acc[j] = gamma[j] + log(theta[j][k]) + log(phi[k][u[t]])
            #   acc[j] = gamma[j] + log(theta[k,j]) + log(phi[k][u[t]])   # tag3
            end
            gamma′[k] = logsumexp(acc)
        end
        gamma .= gamma′
    end
    @logpdf() += logsumexp(gamma)
end

model = hmm_semisup(data["K"], data["V"], data["T"], data["T_unsup"], data["w"], data["z"], data["u"], data["alpha"], data["beta"])

step_size = 0.001
n_steps = 4

include("../infer_turing.jl")

;
