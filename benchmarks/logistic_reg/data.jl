using StatsFuns

function get_data(d=100, n=10_000)
    X = randn(d, n)
    w = randn(d)
    y = Int.(logistic.(X' * w) .> 0.5)

    return Dict(
        "X" => copy(X'),
        "y" => y,
        "D" => d,
        "N" => n,
    )
end
