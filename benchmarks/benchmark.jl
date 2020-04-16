using DrWatson
@quickactivate "TuringExamples"

results_fn = "results"

models = [
    "high_dim_gauss",
    "gauss_unknown",
    "h_poisson",
     "hmm_semisup",
    "naive_bayes",
    "logistic_reg",
    "sto_volatility",
    "lda",
]

# TODO: support pass in a list of models
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

let buffer=IOBuffer()   # use IOBuffer to deplay writing to the file in the end
    for model in models
        write(buffer, "---\n")
        write(buffer, "$model\n")
        write(buffer, "---\n")
        for ppl in ppls
            write(buffer, "[$ppl]\n")
            @info "Benchmarking $model using $ppl ..."
            cmd = `julia $(projectdir("benchmarks", model, "$ppl.jl")) --benchmark`
            if "WANDB" in keys(ENV) && ENV["WANDB"] == "1"
                # Logging to W&B
                withenv("MODEL_NAME" => model) do
                    res = read(cmd, String)
                end
            else
                res = read(cmd, String)
            end
            write(buffer, res)
        end
        write(buffer, "\n")
    end
    # Write to file
    results = take!(buffer)
    open(projectdir("benchmarks", "$results_fn.txt"), "w") do file
        write(file, results)
    end
end