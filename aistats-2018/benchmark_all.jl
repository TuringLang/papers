for model_name in [
    "high-dim-gauss",
    # "naive-bayes",
    "logistic-reg",
]
    println("Benchmarking $model_name")
    println("[Turing]")
    run(`julia $model_name/turing.jl --benchmark`)
    println("[Stan]")
    run(`julia $model_name/stan.jl --benchmark`)
end