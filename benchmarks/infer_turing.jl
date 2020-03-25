using ReverseDiff, Memoization, Zygote
Turing.setadbackend(:reversediff)
Turing.setcache(true)

alg = HMC(step_size, n_steps)

n_samples = 2_000

chain = nothing

using BenchmarkTools

const ADBACKENDS = Dict(
    "forwarddiff" => Turing.Core.ForwardDiffAD{40},
    "reversediff" => Turing.Core.ReverseDiffAD{true},
    "tracker" => Turing.Core.TrackerAD,
    # "zygote" => Turing.Core.ZygoteAD,
)

function get_eval_functions(step_size, n_steps, model)
    spl_prior = Turing.SampleFromPrior()
    function forward_model(x)
        vi = Turing.VarInfo(model)
        vi[spl_prior] = x
        model(vi, spl_prior)
        Turing.getlogp(vi)
    end
    grad_funcs = map(values(ADBACKENDS)) do adbackend
        alg_ad = HMC{adbackend}(step_size, n_steps)
        vi = Turing.VarInfo(model)
        spl = Turing.Sampler(alg_ad, model)
        Turing.Core.link!(vi, spl)
        x -> Turing.Core.gradient_logp(adbackend(), x, vi, model, spl)
    end
    x = Turing.VarInfo(model)[Turing.SampleFromPrior()]
    return x, forward_model, grad_funcs
end

theta, forward_model, grad_funcs = get_eval_functions(step_size, n_steps, model)

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
    t_forward = @belapsed $forward_model($theta)
    println("  Forward time: $t_forward")
    if clog
        wandb.run.summary.time_mean = t_mean
        wandb.run.summary.time_std = t_std
        wandb.run.summary.time_forward = t_forward
    end
    for (name, grad_func) in zip(keys(ADBACKENDS), grad_funcs)
        t = @belapsed $grad_func($theta)
        println("  Gradient time ($name): $t")
        if clog
            s = Symbol("time_gradient_$name")
            eval(:(wandb.run.summary.$s=$t))
        end
    end
elseif "--function" in ARGS
    @btime $forward_model($theta)
    for grad_func in grad_funcs
        @btime $grad_func($theta)
    end
else
    @time chain = sample(model, alg, n_samples; progress_style=:plain)
end
