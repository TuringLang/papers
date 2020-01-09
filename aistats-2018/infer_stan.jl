using CmdStan

model = Stanmodel(
    model=model_str, 
    nchains=1,
    Sample(
        algorithm=CmdStan.Hmc(
            CmdStan.Static(0.4),
            CmdStan.diag_e(),
            0.1,
            0.0,
        ),
        num_warmup=0,
        num_samples=2_000,
        adapt=CmdStan.Adapt(engaged=false),
        save_warmup=true,
    ),
    printsummary=false,
    output_format=:array
)

@time status, chain = stan(model, data, summary=false)
