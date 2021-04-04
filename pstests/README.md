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
python run_local.py --model wdl_criteo (--all) (--val)
```

### PS mode execution
Run ps locally in one terminal (the test file automatically launch scheduler, server and workers):
```
python run_ps_local.py --model wdl_criteo (--all) (--val) (--cache lfuopt) (--bound 10)
```
You can also specify the cache to be used and also the cache bound.

If you would like to run in distribution, please launch scheduler and server in one terminal:
```
python launch_schevers.py (--config xxxx)
```
And launch workers one by one in other terminals:
```
python launch_worker.py --model wdl_criteo (--config xxxx) ...
```
If you define more than one workers in a single configuration file, it's also possible to launch them all:
```
python run_ps_local.py --model wdl_criteo (--config xxxx) ...
```
This file simply launches all the node in the configuration file. 


### Hybrid mode execution
You must launch a scheduler and server in one terminal:
```
python launch_schevers.py (--config xxxx)
```
And then launch the workers simultaneously using mpirun command:
```
../build/_deps/openmpi-build/bin/mpirun -np 8 --allow-run-as-root python run_hybrid_local.py --model wdl_criteo ...
```
Or if in distributed nodes setting:
```
../build/_deps/openmpi-build/bin/mpirun -mca btl_tcp_if_include \[network card name or ip\] -x NCCL_SOCKET_IFNAME=\[network card name\] --host \[host ips\] --allow-run-as-root python run_hybrid_local.py --model wdl_criteo ...
```


### Configuration files
We accept YAML(or JSON) files. We recommend YAML since in YAML we can define several tasks, while in JSON we only define one. Here's an example of YAML file:
```
shared: &shared
  DMLC_PS_ROOT_URI : 127.0.0.1
  DMLC_PS_ROOT_PORT : 13200
  DMLC_NUM_WORKER : 2
  DMLC_NUM_SERVER : 1
  DMLC_PS_VAN_TYPE : p3
sched:
  <<: *shared
  DMLC_ROLE : scheduler
s0:
  <<: *shared
  DMLC_ROLE : server
  SERVER_ID : 0
  DMLC_PS_SERVER_URI : 127.0.0.1
  DMLC_PS_SERVER_PORT : 13201
w0:
  <<: *shared
  DMLC_ROLE : worker
  WORKER_ID : 0
  DMLC_PS_WORKER_URI : 127.0.0.1
  DMLC_PS_WORKER_PORT : 13210
w1:
  <<: *shared
  DMLC_ROLE : worker
  WORKER_ID : 1
  DMLC_PS_WORKER_URI : 127.0.0.1
  DMLC_PS_WORKER_PORT : 13222
  ```

This part is from ./settings/local_s1_w2.yml, where we define a scheduler, a server and two workers. This file is used for run_ps_local.py. For launch_schevers.py, please use a file containing only scheduler and server; for launch_worker.py, please use a single json file; for run_ps_local.py or run_hybrid_local.py, please use a yaml file containing several workers. 

To simplify the execution, you can also omit the uri and port for workers and servers, but only remain the information of its role and the root. The simplified configurations are in ../geometric/config directory, which serve as examples. You can also use ```python -m athena.launcher \[configuraion\] -n \[server#\] (--sched)``` to launch several servers and scheduler.
