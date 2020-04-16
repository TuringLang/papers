using Distributions

function get_data(
    v=100,      # words
    k=5,        # topics
    m=10,       # docs
    d=1_000,    # avg doc length
)
    alpha = ones(k)
    beta = ones(v)

    phi = rand(Dirichlet(beta), k)
    theta = rand(Dirichlet(alpha), m)
    doc_length = rand(Poisson(d), m)

    n = sum(doc_length)

    w = Vector{Int}(undef, n)
    doc = Vector{Int}(undef, n)

    n = 0
    for m = 1:m, i=1:doc_length[m]
        n = n + 1
        z = rand(Categorical(theta[:,m]))
        w[n] = rand(Categorical(phi[:,z]))
        doc[n] = m
    end

    return Dict(
        "K" => k,
        "V" => v,
        "M" => m,
        "N" => n,
        "w" => w,
        "doc" => doc,
        "alpha" => alpha,
        "beta" => beta
    )
end