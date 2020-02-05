Turing.setadbackend(:reverse_diff)

alg = HMC(step_size, n_steps)

n_samples = 2_000

chain = nothing

using BenchmarkTools

function get_eval_functions(model)
    vi = Turing.VarInfo(model)
    spl = Turing.SampleFromPrior()
    # model(vi)
    Turing.Core.link!(vi, spl)
	function forward_model(x)
		vi[spl] = x
		model(vi)
		Turing.getlogp(vi)
    end
    function gradient_forwarddiff(x)
        Turing.Core.gradient_logp_forward(x, vi, model)
    end
    function gradient_tracker(x)
        Turing.Core.gradient_logp_reverse(Turing.Core.TrackerAD(), x, vi, model)
    end
    function gradient_zygote(x)
        Turing.Core.gradient_logp_reverse(Turing.Core.ZygoteAD(), x, vi, model)
    end
    return vi[spl], forward_model, gradient_forwarddiff, gradient_tracker, gradient_zygote
end

theta, forward_model, gradient_forwarddiff, gradient_tracker, gradient_zygote = get_eval_functions(model)

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
    t_gradient_forwarddiff = @belapsed $gradient_forwarddiff($theta)
    println("  Gradient time (ForwardDiff): $t_gradient_forwarddiff")
    t_gradient_trakcer = @belapsed $gradient_tracker($theta)
    println("  Gradient time (Tracker): $t_gradient_trakcer")
    t_gradient_zygote = @belapsed $gradient_zygote($theta)
    println("  Gradient time (Zygote): $t_gradient_zygote")
    if clog
        wandb.run.summary.time_mean                 = t_mean
        wandb.run.summary.time_std                  = t_std
        wandb.run.summary.time_forward              = t_forward
        wandb.run.summary.time_gradient_forwarddiff = t_gradient_forwarddiff
        wandb.run.summary.time_gradient_trakcer     = t_gradient_trakcer
        wandb.run.summary.time_gradient_zygote      = t_gradient_zygote
    end
elseif "--functions" in ARGS
    @btime $forward_model($theta)
    @btime $gradient_forwarddiff($theta)
    @btime $gradient_tracker($theta)
    @btime $gradient_zygote($theta)
else
    @time chain = sample(model, alg, n_samples; progress_style=:plain)
end
