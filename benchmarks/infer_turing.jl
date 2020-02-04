alg = HMC(step_size, n_steps)

n_samples = 2_000

chain = nothing

using BenchmarkTools

varinfo = Turing.VarInfo(model)
spl = Turing.SampleFromPrior()
Turing.Core.link!(varinfo, spl)
forward_model() = model(varinfo, spl)

if "--benchmark" in ARGS
    using Logging: with_logger, NullLogger
    using Statistics: mean, std
    clog = "MODEL_NAME" in keys(ENV)    # cloud logging flag
    if clog
        # Setup W&B
        using PyCall: pyimport
        wandb = pyimport("wandb")
        wandb.init(project="turing-benchmark")
        wandb.config.update(Dict("ppl" => "turing", "model" => ENV["MODEL_NAME"]))
    end
    n_runs = 3
    times = []
    for i in 1:n_runs+1
        with_logger(NullLogger()) do    # disable numerical error warnings
            t = @elapsed sample(model, alg, n_samples; progress=false, raw_output=true)
            clog && i > 1 && wandb.log(Dict("time" => t))
            push!(times, t)
        end
    end
    t_mean = mean(times[2:end])
    t_std = std(times[2:end])
    # Estimate compilation time
    t_with_compilation = times[1]
    t_compilation_approx = t_with_compilation - t_mean
    println("Benchmark results")
    println("  Compilation time: $t_compilation_approx (approximately)")
    println("  Running time: $t_mean +/- $t_std ($n_runs runs)")
    t_forward = @belapsed forward_model()
    println("  Forward time: $t_forward")
    if clog
        wandb.run.summary.time_mean    = t_mean
        wandb.run.summary.time_std     = t_std
        wandb.run.summary.time_forward = t_forward
    end
elseif "--forward_only" in ARGS
    @btime forward_model()
else
    @time chain = sample(model, alg, n_samples; progress_style=:plain)
end
