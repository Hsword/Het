##  Prepare graph datasets

1. Prepare ogbn-mag dataset: ogbn-mag is downloaded from Open Graph Benchmark, so you will have to install ogb.

   ```
   conda install -c conda-forge ogb
   ```
then you can use the following command to partition the graph into 4 parts for 4-workers to use.
   ```
   python3 geometric/utils/part_graph.py -d ogbn-mag --sparse -n 4 -p ~/yourDataPath
   ```
Also note that if you want to train on K node, replace the -n 4 with -n K.
2. Prepare Reddit dataset: We download Reddit dataset from Pytorch geometric(Pyg), so you will have to install pyg. [This Page](https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html) tells you how to install pyg. After pyg is ready, still use the part_graph.py to partition the graph.
   ```
   python3 geometric/utils/part_graph.py -d Reddit --sparse -n 4 -p ~/yourDataPath
   ```

3. Prepare Amazon dataset: This dataset is introduced in the cluster-GCN paper and there are two file to be downloaded: [metadata.json](https://drive.google.com/file/d/0B2jJQxNRDl_rVVZCdWVnYmUyRDg) and [map_files](https://drive.google.com/file/d/0B3lPMIHmG6vGd2U3VHB0Wkk4cGM). Once you download and extract the files and put them together under geometrc/utils/ directory you can run

   ```
   python3 geometric/utils/prepare_amazon_dataset.py
   ```

   Note that you need nltk installed in your environment to run this script and this will take a while.

   After running the script, you will get the two output file: graph.npz and sparsefeature.npy. Put them in the right place.

   ```
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
python3 geometric/tests/test_sparse.py configfile -p ~/yourDataPath/Reddit
```

To train on Hybrid communication mode. Run

```
build/_deps/openmpi-build/bin/mpirun -np 4 --allow-run-as-root python3 test_sparse_hybrid.py configfile -p ~/yourDataPath/Reddit
```

When running on Hybrid mode, you will also have to launch some servers and scheduler seperately

```
python3 -m athena.launcher configfile -n 1 --sched
```

