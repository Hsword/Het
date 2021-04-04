#include "common.h"

#include "graph.h"
#include "sampler.h"
#include "cache.h"
#include "utils.h"

PYBIND11_MODULE(libc_GNN, m) {
  m.doc() = "GNN C++ plugin"; // optional module docstring

  py::class_<PyGraph, std::shared_ptr<PyGraph>>(m, "Graph")
    .def(py::init(&makeGraph), py::arg("x"), py::arg("y"), py::arg("edge_index"), py::arg("num_classes"))
    .def_property_readonly("x", &PyGraph::getX)
    .def_property_readonly("y", &PyGraph::getY)
    .def_property_readonly("edge_index", &PyGraph::getEdgeIndex)
    .def_property_readonly("num_nodes", &PyGraph::nNodes)
    .def_property_readonly("num_edges", &PyGraph::nEdges)
    .def_property_readonly("num_features", &PyGraph::nFeatures)
    .def_property_readonly("num_classes", &PyGraph::nClasses)
    .def("part_graph", &PyGraph::part_graph, py::arg("nparts"), py::arg("balance_edge")=true)
    .def("partition", &PyGraph::PyPartition)
    .def("gcn_norm", &PyGraph::gcnNorm)
    .def("add_self_loop", &PyGraph::addSelfLoop)
    .def("remove_self_loop", &PyGraph::removeSelfLoop)
    .def("__repr__", [](PyGraph &g) {
          std::stringstream ss;
          ss << "<PyGraph Object, nodes=" << g.nNodes() << ",";
          ss << "edges=" << g.nEdges() << ">";
          return ss.str();
        });

  py::class_<PyCache>(m, "Cache")
    .def(py::init(&makeCache))
    .def_property_readonly("limit", &PyCache::getLimit)
    .def("size", &PyCache::getSize)
    .def("insert", &PyCache::insertItem)
    .def("lookup", &PyCache::queryItem)
    .def("lookup_packed", &PyCache::queryItemPacked)
    .def("keys", &PyCache::getKeys);

  py::class_<DistributedSampler>(m, "DistributedSampler")
    .def(py::init(&makeSampler))
    .def_property_readonly("rank", &DistributedSampler::getRank)
    .def_property_readonly("indptr", &DistributedSampler::getIndptr)
    .def_property_readonly("indices", &DistributedSampler::getIndices)
    .def_property_readonly("nodes_from", &DistributedSampler::getNodesFrom)
    .def_property_readonly("local_degree", &DistributedSampler::getLocalDegree)
    .def("sample_neighbors", &DistributedSampler::sampleNeighbours)
    .def("generate_local_subgraph", &DistributedSampler::generateLocalGraph);

  m.def_submodule("utils")
    .def("random_index", &randomIndex)
    .def("construct_graph", &constructGraph)
    .def("sample_head", &sampleHead);
}
