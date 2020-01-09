models = [
    "high-dim-gauss",
    "naive-bayes",
    "logistic-reg",
]

for m in models
    if "--$(m)_only" in models
        models = [m]
        break
    end
end

ppls = [
    "turing", 
    "stan",
]

for p in ppls
    if "--$(p)_only" in ppls
        ppls = [p]
        break
    end
end

open("results.txt", "w") do io
    for model in models
        write(io, "---\n")
        write(io, "Benchmarking $model\n")
        for ppl in ppls
            write(io, "[$ppl]\n")
            res = read(`julia $model/$ppl.jl --benchmark`, String)
            write(io, res)
        end
    end
end