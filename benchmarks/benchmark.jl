results_fn = "results"

models = [
    "high-dim-gauss",
    "gauss-unknown",
    "h_poisson",
    "naive-bayes",
    "logistic-reg",
]

for m in models
    if "--$(m)-only" in ARGS
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
    if "--$(p)-only" in ARGS
        global ppls, results_fn
        ppls = [p]
        results_fn *= "-$(p)"
        break
    end
end

@info "Benchmark config" results_fn models ppls

open("$results_fn.txt", "w") do io
    for model in models
        write(io, "---\n")
        write(io, "$model\n")
        write(io, "---\n")
        for ppl in ppls
            write(io, "[$ppl]\n")
            @info "Benchmarking $model using $ppl ..."
            cmd = `julia $model/$ppl.jl --benchmark`
            if "WANDB" in keys(ENV) && ENV["WANDB"] == "1"
                withenv("MODEL_NAME" => model) do
                    res = read(cmd, String)
                end
            else
                res = read(cmd, String)
            end
            write(io, res)
        end
        write(io, "\n")
    end
end