# Runtime

## Train with Hybrid Parallel

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


## Train with Switch Parallel (SP) Mechanism

Run the following commands on each worker:
```bash
SP_EPOCH=60
CHECKPOINT_DIR=<path to directory saving checkpoints>

python3 main.py --data_dir <path to dataset> -m models.alexnet --config_path models/alexnet/dp_conf.json -o models/alexnet/optim_config.json --no_input_pipelining --end_epoch $SP_EPOCH --checkpoint_dir $CHECKPOINT_DIR --master_addr <IP address of node 0> --rank <worker rank> --local_rank <worker local rank>

python3 main.py --data_dir <path to dataset> -m models.alexnet --config_path models/alexnet/hybrid_conf.json -o models/alexnet/optim_config.json --start_epoch $SP_EPOCH --resume $CHECKPOINT_DIR --master_addr <IP address of node 0> --rank <worker rank> --local_rank <worker local rank>
```
