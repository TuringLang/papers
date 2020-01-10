using Distributions

function get_data(k=5, v=20, t=100, t_unsup=200)
    # Hyper parameters
    alpha = fill(1, k)
    beta = fill(0.1, v)

    # Init containers
    w = Vector{Int}(undef, t)
    z = Vector{Int}(undef, t)
    u = Vector{Int}(undef, t_unsup)

    # Parameters
    theta = rand(Dirichlet(alpha), k)
    phi = rand(Dirichlet(beta), k)

    # Simulate data

    # Supervised
    if t > 0
        z[1] = rand(1:k)
    end
    for t in 2:t
        z[t] = rand(Categorical(theta[:,z[t-1]]))
    end
    for t in 1:t
        w[t] = rand(Categorical(phi[:,z[t]]))
    end

    # Unsupervised
    y = Vector{Int}(undef, t_unsup)
    y[1] = rand(1:k)
    for t in 2:t_unsup
        y[t] = rand(Categorical(theta[:,y[t-1]]))
    end
    for t in 1:t_unsup
        u[t] = rand(Categorical(phi[:,y[t]]))
    end

    return Dict(
        "K" => k,
        "V" => v,
        "T" => t,
        "T_unsup" => t_unsup,
        "w" => w,
        "z" => z,
        "u" => u,
        "alpha" => alpha,
        "beta" => beta
    )
end