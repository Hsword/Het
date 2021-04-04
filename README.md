# HET

[![license](https://img.shields.io/github/license/apache/zookeeper?color=282661)](https://github.com/Hsword/Het/blob/main/LICENSE)

A distributed deep learning framework for huge embedding model training (previouly named Athena). HET is developed by <a href="http://net.pku.edu.cn/~cuibin/" target="_blank" rel="nofollow">DAIM Lab</a> at Peking University. This is a previewed version for the reviewers to verify our reproducibility and the whole system is not fully released. If you have any questions, please email to *xupeng.miao@pku.edu.cn*

## Installation
1. Clone the respository.
2. Edit the athena.exp file and set the environment path for python.

```bash
source athena.exp
```

3. CMake is used to compile Hetu. Generate the Makefile first:
```bash
conda install cmake # ensure cmake version >= 3.18
cp cmake/config.example.cmake cmake/config.cmake
# modify paths for CUDA, CUDNN, NCCL, MPI in cmake/config.cmake if necessary
mkdir build && cd build && cmake ..
# if nccl needed, please download nccl 2.7.8 and install.
# if hetu cache needed, please install pybind11: conda install pybind11.
# if GNN needed, please install metis.
```

4. Compile Athena by Makefile
```bash
# current directory is ./build/
make clean
make athena version=mkl -j 32
make athena version=gpu -j 32
# or: make athena version=all -j 32
make ps pslib -j 32 # for ps support
make mpi mpi_nccl -j 32 # for mpi-based allreduce, time-consuming
# btw: make -j32 does all the things
```

5. Install graphviz to support graph board visualization (not maintained, may deprecate)
```bash
sudo apt-get install graphviz
sudo pip install graphviz
```

6. Run some simple examples

Train logistic regression with gpu:

```bash
python tests/models_tests/main.py --model logreg --validate
```

Train a 3-layer mlp with cpu:

```bash
python tests/models_tests/main.py --model mlp --validate --gpu -1
```

Train a 3-layer mlp with gpu:

```bash
python tests/models_tests/main.py --model mlp --validate
```

Train a 3-layer cnn with cpu:

```bash
python tests/models_tests/main.py --model cnn_3_layers --validate --gpu -1
```

Train a 3-layer cnn with gpu:

```bash
python tests/models_tests/main.py --model cnn_3_layers --validate
```

Train a 3-layer mlp with allreduce on 2 gpus (use mpirun in open-mpi path):
```bash
path/to/deps/mpirun --allow-run-as-root -np 2 python tests/models_tests/allreduce_main.py --model mlp --validate
```

Train a 3-layer mlp with PS on 1 server and 2 workers (need to set configurations in json files):
```bash
# in scheduler process
python tests/models_tests/ps_main.py --model mlp --setting scheduler_conf.json
# in server process
python tests/models_tests/ps_main.py --model mlp --setting server_conf.json
# in worker1 process
python tests/models_tests/ps_main.py --model mlp --setting worker_conf.json --validate
# in worker2 process
python tests/models_tests/ps_main.py --model mlp --setting worker_conf_2.json --validate
```

Graphboard is on http://localhost:9997 during training. The port can be changed by the PORT of mnist_dlsys.py. (not maintained, may deprecate)


## Evaluation on CTR and GNN tasks: 
Please refer to our [examples](het_examples/).

## License

The entire codebase is under [Apache-2.0 license](LICENSE)
