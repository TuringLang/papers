function get_data(n=10_000)
    m = randn()
    s = abs(rand()) + 0.5
    y = m .+ s * randn(n)
    return Dict(
        "N" => n,
        "y" => y
        )
end
