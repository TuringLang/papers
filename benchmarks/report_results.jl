using ArgParse

s = ArgParseSettings()

@add_arg_table s begin
    "filename"
        help = "CSV file name"
        required = true
end

parsed_args = parse_args(s)
filename = parsed_args["filename"]

using PyCall
pd = pyimport("pandas")
tabulate = pyimport("tabulate")
tabulate.PRESERVE_WHITESPACE = true

model_dict = Dict(
    "gauss_unknown"     => "Gaussian Unknown",
    "h_poisson"         => "Hierarchical Poisson",
    "high_dim_gauss"    => "10,000D Gaussian",
    "hmm_semisup"       => "Semi-Supervised HMM",
    "lda"               => "LDA",
    "logistic_reg"      => "Logistic Regression",
    "naive_bayes"       => "Naive Bayes",
    "sto_volatility"    => "Stochastic Volatility",
)

# Read CSV
df = pd.read_csv(filename)
# Rename models
df."model" = df."model".replace(model_dict)
# Capitalise
df = df.rename(columns=Dict("model" => "Model", "ppl" => "PPL"))
# Round precision
df = df.round(Dict("time_mean" => 3, "time_std" => 3))
# Create mean +/ std
df_time = py"$df[['time_mean', 'time_std']].apply(lambda x: ' ± '.join(map(lambda y: y.rjust(6, ' '), map(str, x))), axis=1)"
df_time = df_time.rename("time")
df = pd.concat([df, df_time], axis=1)
# Pivot by model
function postprocess(df)
    # df_ratio = py"$df[['stan', 'turing']].apply(lambda x: float(x[0]) / float(x[1]), axis=1)"
    # df_ratio = df_time.rename("ratio")
    # df = pd.concat([df, df_ratio], axis=1)
    df = df.reindex(
        columns=["turing", "stan"]
    ).rename(
        columns=Dict(
            "stan"   => "Stan",
            "turing" => "Turing",
        )
    )
    return df
end
df_text = df.pivot(index="Model", columns="PPL", values="time") |> postprocess
df_plot = df.pivot(index="Model", columns="PPL", values="time_mean") |> postprocess

# Report table in text
println(replace(string(df_text), "PyObject " => ""))
println(tabulate.tabulate(df_text, tablefmt="pipe", headers="keys"))

# Make visualization
ax = df_plot.plot.bar(rot=0)
ax.set_ylabel("Time (s)")
ax.set_yscale("log")
for tick in ax.get_xticklabels()
    tick.set_rotation(10)
end
fig = ax.get_figure()
fig.set_size_inches(14, 7)
fig.savefig("$(split(filename, ".")[1]).png")
