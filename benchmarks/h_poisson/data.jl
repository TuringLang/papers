# Ref: https://github.com/StatisticalRethinkingJulia/MCMCBenchmarkSuite.jl/blob/master/src/Hierarchical_Poisson/Hierarhical_Poisson_Models.jl

using Distributions: Normal, Poisson

function get_data(nd=5, ns=10, a0=1, a1=0.5, a0_sig=0.3)
    n = nd * ns
    y = zeros(Int, n)
    x = zeros(n)
    idx = similar(y)
    i = 0
    for s in 1:ns
        a0s = rand(Normal(0, a0_sig))
        logpop = rand(Normal(9, 1.5))
        λ = exp(a0 + a0s + a1 * logpop)
        for nd in 1:nd
        i += 1
        x[i] = logpop
        idx[i] = s
        y[i] = rand(Poisson(λ))
        end
    end
    return Dict(
        "y" => y,
        "x" => x,
        "idx" => idx,
        "N" => n,
        "Ns" => ns,
    )
end
