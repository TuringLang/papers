using DrWatson
@quickactivate "TuringExamples"

using Random: seed!
seed!(1)

include("data.jl")

data = get_data()

using Turing

using StatsFuns: logsumexp

@model hmm_semisup(K, V, T, T_unsup, w, z, u, alpha, beta, ::Type{Tv}=Vector{Float64}) where {Tv} = begin
    theta ~ filldist(Dirichlet(alpha), K)
    phi ~ filldist(Dirichlet(beta), K)
    w ~ arraydist(Categorical.(copy.(eachcol(phi[:, z]))))
    z[2:end] ~ arraydist(Categorical.(copy.(eachcol(theta[:, z[1:end-1]]))))
    gamma = log.(phi[u[1], 1:K])
    for t in 2:T_unsup
        gamma = mapreduce(vcat, 1:K) do k
            logsumexp(gamma .+ log.(theta[k, 1:K]) .+ log(phi[u[t], k]))
        end
    end
    @logpdf() += logsumexp(gamma)
end

model = hmm_semisup(data["K"], data["V"], data["T"], data["T_unsup"], data["w"], data["z"], data["u"], data["alpha"], data["beta"])

step_size = 0.001
n_steps = 4

include("../infer_turing.jl")

;
