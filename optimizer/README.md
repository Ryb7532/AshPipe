# Optimizer

## Setup

```bash
g++ -O3 -shared -fPIC `python3 -m pybind11 --includes` cppoptimizer.cpp -o cppoptimizer`python3-config --extension-suffix`
```

## Run model partitioning algorithm

```bash
./run_optimizer.sh
```

(If necessary, rewrite `run_optimizer.sh`.)


## Create hybrid parallel model

```bash
./run_converter.sh
```
and then, specify the number of workers in each stage.

(If necessary, rewrite `run_converter.sh`.)

