using Random: seed!
seed!(1)

include("data.jl")

data = get_data()

using Memoization, Turing

using StatsFuns: logsumexp

@model hmm_semisup(K, V, T, T_unsup, w, z, u, alpha, beta, ::Type{Tv}=Vector{Float64}) where {Tv} = begin
    theta ~ filldist(Dirichlet(alpha), K)
    phi ~ filldist(Dirichlet(beta), K)
    w ~ arraydist(Categorical.(eachcol(phi[:, z])))
    z[2:end] ~ arraydist(Categorical.(eachcol(theta[:, z[1:end-1]])))
    gamma = log.(phi[u[1], 1:K])
    for t in 2:T_unsup
        temp = gamma' .+ log.(theta[1:K, 1:K]) .+ log.(phi[u[t], 1:K])
        gamma = vec(logsumexp(temp, dims=2))
    end
    @logpdf() += logsumexp(gamma)
end

model = hmm_semisup(data["K"], data["V"], data["T"], data["T_unsup"], data["w"], data["z"], data["u"], data["alpha"], data["beta"])

step_size = 0.001
n_steps = 4
test_zygote = false
test_tracker = false

include("../infer_turing.jl")

;
