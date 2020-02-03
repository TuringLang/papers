using Turing

function get_data(n=1_000; m0 = 0.0, s0 = 1.0)

    K = round(Int, log(n))
    k = rand(1:K)
    w = rand(Dirichlet(k, 1.0))
    w ./= sum(w)

    m = map(_ -> m0 + randn()*s0, 1:k)

    y = map(_ -> randn() + m[rand(Categorical(w))], 1:n)
    return Dict(
        "N" => n,
        "y" => y,
        "m" => m
        )
end
