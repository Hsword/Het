## Compile the system

* Install necessary dependencies, we use all the dependencies in a environment.yml file. You can use the following commands to prepare for the python environment.

  ```shell
  conda env create
  conda activate hetu
  source athena.exp # set PYTHONPATH
  ```

* compile the system, checkout the README.md in the main page for more detail.

  ```shell
  # in Athena/
  mkdir build && cd build && cmake .. && make -j8
  ```

## Configuration file explained

We use a simple yaml file to specify the run configuration.

```yaml
shared :
  DMLC_PS_ROOT_URI : 127.0.0.1
  DMLC_PS_ROOT_PORT : 13100
  DMLC_NUM_WORKER : 4
  DMLC_NUM_SERVER : 1
launch :
  worker : 4
  server : 1
  scheduler : true
```

The 4 k-v pair in "shared" are used for PS-lite parameter server and will be added into environment. When running on a cluster, you should change "DMLC_PS_ROOT_URI" into an available IP address in the cluster.

The following "launch" is only used in PS-mode (ommitted in hybrid mode). This means that the number of worker, server and scheduler launched locally on this machine. In hybrid mode, workers are launched by mpirun. Servers and schedulers will be launched by

```shell
python3 -m athena.launcher configfile [-n NumServer] [--sched]
```

Note that there should be only 1 scheduler and should only be launched on the machine with DMLC_PS_ROOT_URI.

Note that the launch automatically select network interface for you. If this fails, try adding "DMLC_INTERFACE : eth0" to select the right network device.

##  Prepare graph datasets

1. Prepare ogbn-mag dataset: ogbn-mag is downloaded from Open Graph Benchmark, and the download may take a while.

   Then you can use the following command to partition the graph into 4 parts for 4-workers to use.

   ```
   python3 geometric/utils/part_graph.py -d ogbn-mag --sparse -n 4 -p ~/yourDataPath
   ```

   Also note that if you want to train on K node, replace the -n 4 with -n K.

2. Prepare Reddit dataset: We download Reddit dataset from Pytorch geometric(Pyg), so you will have to install pyg. Since pyg depends on pytorch version and cuda version, we don't install pyg for you automatically. [This Page](https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html) tells you how to install pyg. After pyg is ready, still use the part_graph.py to partition the graph.

   ```
   python3 utils/part_graph.py -d Reddit --sparse -n 4 -p ~/yourDataPath
   ```

3. Prepare Amazon dataset: This dataset is introduced in the cluster-GCN paper and there are two file to be downloaded: [metadata.json](https://drive.google.com/file/d/0B2jJQxNRDl_rVVZCdWVnYmUyRDg) and [map_files](https://drive.google.com/file/d/0B3lPMIHmG6vGd2U3VHB0Wkk4cGM). Once you download and extract the files and put them together under geometrc/utils/ directory you can run

   ```
   python3 utils/prepare_amazon_dataset.py
   ```

   Note that you need nltk installed in your environment to run this script and this will take a while.

   After running the script, you will get the two output file: graph.npz and sparsefeature.npy. Put them in the right place.

   ```shell
   # at Athena/
   mkdir -p geometric/GNN/dataset/.dataset/AmazonSparseNode
   mv graph.npz sparsefeature.npy geometric/GNN/dataset/.dataset/AmazonSparseNode/
   ```

   Finally, use the part_graph.py to partition the graph

   ```
   python3 geometric/utils/part_graph.py -d AmazonSparseNode --sparse -n 4 -p ~/yourDataPath
   ```

## Training GNN Embedding Models

After you have prepare one of the three graph datasets,you can start training Embedding Models on graph datasets. We take Reddit as an example.

To train on PS communication mode. Run

```
python3 run_sparse.py configfile -p ~/yourDataPath/Reddit
```

To train on Hybrid communication mode. Run

```
mpirun -np 4 --allow-run-as-root python3 run_sparse_hybrid.py configfile -p ~/yourDataPath/Reddit
```

When running on Hybrid mode, you will also have to launch some servers and scheduler seperately

```
python3 -m athena.launcher configfile -n 1 --sched
```

# Tests for distributed CTR models.

## Prepare criteo data

* We have provided a sampled version of kaggle-criteo dataset, which locates in ./datasets/criteo/ . To use the given data, please do not specify the 'all' flag and 'val' flag when running test files.
* To download the original kaggle-criteo dataset, please specify a source in models/load_data.py and use ```python models/load_data.py``` to download the whole kaggle-criteo dataset.


## Flags for test files

Here we explain some of the flags you may use in test files:

* model: to specify the model, candidates are \['wdl_criteo', 'dfm_criteo', 'dcn_criteo'\]
* config: to specify the configuration file in settings.
* val: whether using validation.
* cache: whether using cache in PS/Hybrid mode.
* bsp: whether using bsp (default asp) in PS/Hybrid mode. (In Hybrid, AllReduce can enforce dense parameters to use bsp, so there will be no stragglers.)
* all: whether to use all criteo data.
* bound: per embedding entry staleness in cache setting, default to be 100.


## Local tests

If memory available, you can try to run the model locally, by running

* run_local.py: run CTR models in our system.
* run_ps_local.py: run CTR models using PS locally.
* run_tf_local.py: run CTR models in TF locally.
* run_hybrid_local.py: run CTR models in Hybrid mode locally; need to start up another terminal for schevers (explained below).


## Distributed tests

1. specify settings in ./settings/ directory and launch schecers(scheduler and server): use ```python launch_schevers.py```
2. launch workers:
   - ```python launch_workers.py``` to launch one worker, or ```python run_ps_local.py``` use a configuration file with only workers to use PS mode.
   - ```mpirun ... python hybrid_local.py``` to launch workers in hybrid mode. Please refer to corresponding test file to see the complete command.
   - For distributed TF, please use ```python tf_launch_server.py``` and ```python tf_launch_worker.py```.

## Settings

Or say configurations, located in ./settings/ directory. YAML or JSON accepted.

## Models / TF models

* wdl_adult: wide & deep for adult dataset.
* wdl_criteo: wide & deep for criteo dataset.
* dfm_criteo: deepFM for criteo dataset.
* dcn_criteo: deep & cross for criteo dataset.


## Examples

### Local execution

Run wdl with criteo locally(if the whole dataset is downloaded, you can use all data or use validate data):

```
python3 run_local.py --model wdl_criteo (--all) (--val)
```

### PS mode execution

Run ps locally in one terminal (the test file automatically launch scheduler, server and workers):

```
python3 run_ps_local.py --model wdl_criteo (--all) (--val) (--cache lfuopt) (--bound 10)
```

You can also specify the cache to be used and also the cache bound.

If you would like to run in distribution, please launch scheduler and server in one terminal:

```
python3 -m athena.launcher configfile -n 1 --sched
```

And launch workers on another machine.

```
python3 run_ps_local.py --model wdl_criteo --config configfile ...
```


### Hybrid mode execution

You must launch a scheduler and server in one terminal:

```
python3 -m athena.launcher configfile -n 1 --sched
```

And then launch the workers simultaneously using mpirun command:

```
mpirun -np 8 --allow-run-as-root python run_hybrid_local.py --model wdl_criteo ...
```

Or if in distributed nodes setting:

```
mpirun -mca btl_tcp_if_include \[network card name or ip\] -x NCCL_SOCKET_IFNAME=\[network card name\] --host \[host ips\] --allow-run-as-root python run_hybrid_local.py --model wdl_criteo ...
```
