alg = HMC(0.1, 4)
n_samples = 2_000

chain = nothing

if "--benchmark" in ARGS
    using Logging: with_logger, NullLogger
    using Statistics: mean, std
    n_runs = 3
    times = []
    for i in 1:n_runs+1
        with_logger(NullLogger()) do
            t = @elapsed sample(model, alg, n_samples; progress=false, raw_output=true)
            push!(times, t)
        end
    end
    t_with_compilation = times[1]
    t_mean = mean(times[2:end])
    t_std = std(times[2:end])
    t_compilation_approx = t_with_compilation - t_mean
    println("Benchmark results")
    println("  Compilation time: $t_compilation_approx (approximately)")
    println("  Running time: $t_mean +/- $t_std ($n_runs runs)")
else
    @time chain = sample(model, alg, n_samples; progress_style=:plain)
end