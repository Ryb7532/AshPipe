# AshPipe: Asynchronous Hybrid Pipeline Parallel for DNN Training

This repository cantains AshPipe (ASynchronous Hybrid PIPEline parallel), a hybrid parallel framework that combines data parallelism and asynchronous pipeline parallelism.

Paper: [AshPipe: Asynchronous Hybrid Pipeline Parallel for DNN Training](https://dl.acm.org/doi/10.1145/3635035.3635045)


## Setup

```bash
pip install -r requirements.txt

cd optimizer

g++ -O3 -shared -fPIC `python3 -m pybind11 --includes` cppoptimizer.cpp -o cppoptimizer`python3-config --extension-suffix`
```


## End-to-end Workflow

To run a demo, run the following commands:

[from `./profiler/image_classification`]
```bash
./run_profiler.sh
```

[from `./optimizer`]
```bash
./run_optimizer.sh
./run_converter.sh
```

[from `./runtime`]
On node 0:
```bash
python3 main.py --data_dir <path to dataset> -m models.alexnet --config_path models/alexnet/hybrid_conf.json -o models/alexnet/optim_config.json --master_addr <IP address of node 0> --rank 0 --local_rank 0
python3 main.py --data_dir <path to dataset> -m models.alexnet --config_path models/alexnet/hybrid_conf.json -o models/alexnet/optim_config.json --master_addr <IP address of node 0> --rank 1 --local_rank 1
python3 main.py --data_dir <path to dataset> -m models.alexnet --config_path models/alexnet/hybrid_conf.json -o models/alexnet/optim_config.json --master_addr <IP address of node 0> --rank 2 --local_rank 2
python3 main.py --data_dir <path to dataset> -m models.alexnet --config_path models/alexnet/hybrid_conf.json -o models/alexnet/optim_config.json --master_addr <IP address of node 0> --rank 3 --local_rank 3
```
On node 1:
```bash
python3 main.py --data_dir <path to dataset> -m models.alexnet --config_path models/alexnet/hybrid_conf.json -o models/alexnet/optim_config.json --master_addr <IP address of node 0> --rank 4 --local_rank 0
python3 main.py --data_dir <path to dataset> -m models.alexnet --config_path models/alexnet/hybrid_conf.json -o models/alexnet/optim_config.json --master_addr <IP address of node 0> --rank 5 --local_rank 1
python3 main.py --data_dir <path to dataset> -m models.alexnet --config_path models/alexnet/hybrid_conf.json -o models/alexnet/optim_config.json --master_addr <IP address of node 0> --rank 6 --local_rank 2
python3 main.py --data_dir <path to dataset> -m models.alexnet --config_path models/alexnet/hybrid_conf.json -o models/alexnet/optim_config.json --master_addr <IP address of node 0> --rank 7 --local_rank 3
```


## Publication

AshPipe is published in HPCAsia2024. To cite our work:
```
@inproceedings{hosoki2024ashpipe,
  title={AshPipe: Asynchronous Hybrid Pipeline Parallel for DNN Training},
  author={Hosoki, Ryubu and Endo, Toshio and Hirofuchi, Takahiro and Ikegami, Tsutomu},
  booktitle={Proceedings of the International Conference on High Performance Computing in Asia-Pacific Region (HPCAsia' 2024)},
  pages={117--126},
  year={2024}
}
```

## License

Licensed under the [MIT license](LICENSE).
