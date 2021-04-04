#ifndef GNN_UTILS_H
#define GNN_UTILS_H

#include "common.h"

#include "graph.h"
#include "sampler.h"

/*
    given an array with size, return an array with the same size
    result[i] = sum(&arr[0], &arr[i]) + rand()%arr[i]
*/
py::array_t<long> randomIndex(py::array_t<long> arr);

/*
    helper function to construct sampled subgraph in distributed settings
    sampler_data:
        [
            (x, y, deg, edges, heads_id, heads_from)
            ...
        ]
*/
PyGraph constructGraph(py::list sample_data, PyGraph &graph,
py::array_t<long> indptr_py, py::array_t<long> indices_py,
  py::array_t<long> nodes_from_py, int rank, int nrank);

py::array_t<long> sampleHead(py::array_t<long> indptr, py::array_t<long> indices, size_t required_num);

#endif /* GNN_UTILS_H */