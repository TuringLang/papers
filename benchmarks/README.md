# Light-weight benchmarking scripts

## Usage

### Benchmark

Run all benchmarks
- `julia benchmark.jl`

Benchmark specific model
- E.g. `julia benchmark.jl --lda-only`

Benchmark specific ppl
- E.g. `julia benchmark.jl --turing-only`

You can use `WANDB=1` to enable logging to [W&B](https://app.wandb.ai/), e.g. `WANDB=1 julia benchmark.jl`
- You must setup W&B on your local machine in the same Python environment that PyCall uses.
- Our results are available at [this](https://app.wandb.ai/xukai92/turing-benchmark/table) W&B table.

### Debug

Single run of model
- E.g. `julia lda/turing.jl`
