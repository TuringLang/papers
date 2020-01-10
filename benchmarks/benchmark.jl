results_fn = "results"

models = [
    "high-dim-gauss",
    "gauss-unknown",
    "naive-bayes",
    "logistic-reg",
]

for m in models
    if "--$(m)_only" in ARGS
        global models, results_fn
        models = [m]
        results_fn = "$m/$results_fn"
        break
    end
end

ppls = [
    "turing", 
    "stan",
]

for p in ppls
    if "--$(p)_only" in ARGS
        global ppls, results_fn
        ppls = [p]
        results_fn *= "_$(p)"
        break
    end
end

@info "Benchmark config" results_fn models ppls

open("$results_fn.txt", "w") do io
    for model in models
        write(io, "---\n")
        write(io, "Benchmarking $model\n")
        write(io, "---\n")
        for ppl in ppls
            write(io, "[$ppl]\n")
            res = read(`julia $model/$ppl.jl --benchmark`, String)
            write(io, res)
        end
        write(io, "\n")
    end
end