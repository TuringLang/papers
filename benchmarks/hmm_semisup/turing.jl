using Random: seed!
seed!(1)

include("data.jl")

data = get_data()

using Turing

Turing.setadbackend(:reverse_diff)

using StatsFuns: logsumexp

# FIXME: tag1 + tag2 + tag3 lead to AD error
@model hmm_semisup(w, z, u, alpha, beta, ::Type{R}=Float64) where {R} = begin
    K, V, T, T_unsup = length.((alpha, beta, w, u))

    theta = Vector{Vector{R}}(undef, K)
    for k = 1:K
      theta[k] ~ Dirichlet(alpha)
    end
    # theta ~ Multi(Dirichlet(alpha), K)  # tag1
    phi = Vector{Vector{R}}(undef, K)
    for k = 1:K
      phi[k] ~ Dirichlet(beta)
    end
  
    w ~ ArrayDist(Categorical.(phi[z]))
    z[2:end] ~ ArrayDist(Categorical.(theta[z[1:end-1]]))
    # z[2:end] ~ ArrayDist(Categorical.(theta[:,zi] for zi in z[1:end-1]))    # tag2

    # Forward algorithm
    acc = Vector{R}(undef, K)
    gamma = Matrix{R}(undef, T_unsup, K)
    for k = 1:K
      gamma[1,k] = log(phi[k][u[1]])
    end
    for t = 2:T_unsup, k = 1:K
        for j = 1:K
          acc[j] = gamma[t-1,j] + log(theta[j][k]) + log(phi[k][u[t]])
        #   acc[j] = gamma[t-1,j] + log(theta[k,j]) + log(phi[k][u[t]])   # tag3
        end
        gamma[t,k] = logsumexp(acc)
    end
    @logpdf() += logsumexp(gamma[T_unsup,:])
end

model = hmm_semisup(data["w"], data["z"], data["u"], data["alpha"], data["beta"])

alg = HMC(0.001, 4)

include("../infer_turing.jl")

;
