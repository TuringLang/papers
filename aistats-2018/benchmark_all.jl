open("results.txt", "w") do io
    for model_name in [
        "high-dim-gauss",
        "naive-bayes",
        "logistic-reg",
    ]
        write(io, "Benchmarking $model_name")
        for ppl in ["turing", "stan"]
            write(io, "[$ppl]")
            res = read(`julia $model_name/$ppl.jl --benchmark`, String)
            write(io, res)
        end
    end
end