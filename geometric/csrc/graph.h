#ifndef GNN_GRAPH_H
#define GNN_GRAPH_H

#include <metis.h>

#include "common.h"

class PyGraph {
private:
  std::vector<float> x_;
  std::vector<int> y_;
  std::vector<long> edge_index_u_, edge_index_v_;
  int num_classes_;
public:
  PyGraph(std::vector<float> &&x, std::vector<int> &&y, std::vector<long> &&edge_index_u,
  std::vector<long> &&edge_index_v, int num_classes);
  ~PyGraph() {}
  PyGraph(const PyGraph &) = delete;
  PyGraph(PyGraph &&);

  //Getter
  auto nNodes() {return y_.size();}
  auto nFeatures() {return x_.size() / nNodes();}
  auto nEdges() {return edge_index_u_.size();}
  auto nClasses() {return num_classes_;}
  auto getX() {
    auto x = V2ANOCOPY(x_);
    x.resize({nNodes(), nFeatures()});
    return x;
  }
  auto getY() {return V2ANOCOPY(y_);}
  auto getEdgeIndex() {return py::make_tuple(V2ANOCOPY(edge_index_u_), V2ANOCOPY(edge_index_v_));}

  //Graph common API
  void addSelfLoop();
  void removeSelfLoop();
  double denseEfficiency();
  std::vector<long> degree();
  py::array_t<double> gcnNorm(bool use_original_gcn_norm);

  //Graph Partition API
  py::tuple part_graph(int nparts, bool balance_edge);
  std::vector<idx_t> partition(idx_t nparts, bool balance_edge);
  py::array_t<idx_t> PyPartition(idx_t nparts);
  friend PyGraph constructGraph(py::list, PyGraph &, py::array_t<long>, py::array_t<long>, py::array_t<long>, int, int);
};

PyGraph makeGraph(py::array_t<float> x, py::array_t<int> y, py::array_t<long> edge_index, int num_classes);

#endif /* GNN_GRAPH_H */
