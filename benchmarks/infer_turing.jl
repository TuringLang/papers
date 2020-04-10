n_samples = 2_000

chain = nothing

using BenchmarkTools
using ReverseDiff, Memoization, Zygote
using Logging: with_logger, NullLogger

function get_eval_functions(step_size, n_steps, model)
    function forward_model(x)
        spl = Turing.SampleFromPrior()
        vi = Turing.VarInfo(model)
        vi[spl] = x
        model(vi, spl)
        Turing.getlogp(vi)
    end
    adbackends = if test_zygote && test_tracker
        (Turing.Core.ForwardDiffAD{40}, Turing.Core.ReverseDiffAD{true},
        Turing.Core.TrackerAD, Turing.Core.ZygoteAD)
    elseif test_tracker
        (Turing.Core.ForwardDiffAD{40}, Turing.Core.ReverseDiffAD{true},
        Turing.Core.TrackerAD)
    else
        (Turing.Core.ForwardDiffAD{40}, Turing.Core.ReverseDiffAD{true})
    end
    funcs = map(adbackends) do adbackend
        alg = HMC{adbackend}(step_size, n_steps)
        vi = Turing.VarInfo(model)
        spl = Turing.Sampler(alg, model)
        Turing.Core.link!(vi, spl)
        x -> Turing.Core.gradient_logp(adbackend(), x, vi, model, spl)
    end
    x = Turing.VarInfo(model)[Turing.SampleFromPrior()]

    return (x, forward_model, funcs...,)
end

if test_zygote && test_tracker
    theta, forward_model, gradient_forwarddiff, gradient_reversediff, gradient_tracker, gradient_zygote = get_eval_functions(step_size, n_steps, model)
elseif test_tracker
    theta, forward_model, gradient_forwarddiff, gradient_reversediff, gradient_tracker = get_eval_functions(step_size, n_steps, model)
else
    theta, forward_model, gradient_forwarddiff, gradient_reversediff = get_eval_functions(step_size, n_steps, model)
end

if "--benchmark" in ARGS
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
    alg = HMC(step_size, n_steps)
    for i in 1:n_runs+1
        with_logger(NullLogger()) do    # disable numerical error warnings
            t = @elapsed sample(model, alg, n_samples; progress=false, chain_type=Any)
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
    t_gradient_forwarddiff = @belapsed $gradient_forwarddiff($theta)
    println("  Gradient time (ForwardDiff): $t_gradient_forwarddiff")
    t_gradient_reversediff = @belapsed $gradient_reversediff($theta)
    println("  Gradient time (ReverseDiff): $t_gradient_reversediff")
    if test_tracker
        t_gradient_trakcer = @belapsed $gradient_tracker($theta)
        println("  Gradient time (Tracker): $t_gradient_trakcer")
    end
    if test_zygote
        t_gradient_zygote = @belapsed $gradient_zygote($theta)
        println("  Gradient time (Zygote): $t_gradient_zygote")
    end
    if clog
        wandb.run.summary.time_mean                 = t_mean
        wandb.run.summary.time_std                  = t_std
        wandb.run.summary.time_forward              = t_forward
        wandb.run.summary.time_gradient_forwarddiff = t_gradient_forwarddiff
        if test_tracker
            wandb.run.summary.time_gradient_trakcer     = t_gradient_trakcer
        end
        if test_zygote
            wandb.run.summary.time_gradient_zygote      = t_gradient_zygote
        end
    end
elseif "--function" in ARGS
    @btime $forward_model($theta)
    @btime $gradient_forwarddiff($theta)
    @btime $gradient_reversediff($theta)
    if test_tracker
        @btime $gradient_tracker($theta)
    end
    if test_zygote
        @btime $gradient_zygote($theta)
    end
else
    alg = HMC(step_size, n_steps)
    with_logger(NullLogger()) do
        @time chain = sample(model, alg, n_samples; progress_style=:plain, chain_type=Any)
    end
end
