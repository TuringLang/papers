model = Stanmodel(
    model=model_str, 
    nchains=1,
    Sample(
        algorithm=alg,
        num_warmup=0,
        num_samples=2_000,
        adapt=CmdStan.Adapt(engaged=false),
        save_warmup=true,
    ),
    printsummary=false,
    output_format=:array
)

if "--benchmark" in ARGS
    using Statistics: mean, std
    n_runs = 3
    times = []
    for i in 1:n_runs
        status, chain = stan(model, data, summary=false)
        tl = read(pipeline(`tail tmp/noname_run.log`, `rg "s \(S"`), String)
        t = parse(Float64, match(r"[0-9]+.[0-9]+", tl).match)
        push!(times, t)
    end
    t_mean = mean(times)
    t_std = std(times)
    println("Benchmark results")
    println("  Running time: $t_mean +/- $t_std ($n_runs runs)")
else
    @time status, chain = stan(model, data, summary=false)
end