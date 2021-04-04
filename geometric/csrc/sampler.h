#ifndef GNN_SAMPLER_H
#define GNN_SAMPLER_H

#include "common.h"

#include <random>

class DistributedSampler {
private:
  std::vector<long> indptr_;
  std::vector<long> indices_;
  std::vector<long> nodes_from_;
  std::vector<long> local_degree_;
  size_t rank_;
  // sortIndices put all the local edges of a node before other outward edges
  void sortIndices();
public:
  DistributedSampler(std::vector<long> &&indptr, std::vector<long> &&indices,
                     std::vector<long> &&nodes_from, int rank);
  DistributedSampler(DistributedSampler &&);
  DistributedSampler(DistributedSampler &) = delete;
  /*
    Given a list of node from the local subgraph, return a random neighbor for each node in the list
    args:
      node_ids, nodes to sample
      ratio : 0<=ratio<=1, nodes will be forced to choose local neighbor with a change ratio
  */
  py::tuple sampleNeighbours(py::array_t<long> node_ids, double ratio = 0);
  /*
    Given some node in the local graph, return all the corresponding edges related to the graph
    Note that all the edges comes from the local subgraph
    args:
      node_ids: an array containing node ids
    returns:
      edges in coo-format
  */
  py::tuple generateLocalGraph(py::array_t<long> node_ids);
  auto getRank() {return rank_;}
  auto getIndptr() {return V2ANOCOPY(indptr_);}
  auto getIndices() {return V2ANOCOPY(indices_);}
  auto getNodesFrom() {return V2ANOCOPY(nodes_from_);}
  auto getLocalDegree() {return V2ANOCOPY(local_degree_);}
  auto nLocalNodes() {return indptr_.size() - 1;}
};

/*
  factory function for sampler
  args:
    edges: a list of length nparts (number of subgraphs), each containing (2, e) elements
    indicating an adjacency matrix in coo-format, which is the edges from local subgraph to i'th subgraph
    rank: local rank
    num_nodes: number of nodes in the local subgraph
*/
DistributedSampler makeSampler(py::list edges, size_t rank, size_t num_nodes);

#endif /* GNN_SAMPLER_H */