#include "utils.h"

#include <map>

py::array_t<long> randomIndex(py::array_t<long> arr) {
  assert(arr.ndim() == 1);
  size_t n = arr.size();
  auto arr_c = arr.data();
  std::vector<long> result;
  {
    py::gil_scoped_release release;
    size_t ac = 0;
    for (size_t i = 0; i < n; i++) {
      result.push_back(ac + rand()%arr_c[i]);
      ac += arr_c[i];
    }
  }
  return V2A(result);
}

py::array_t<long> sampleHead(py::array_t<long> indptr, py::array_t<long> indices, size_t required_num) {
  size_t nnodes = indptr.size() - 1;
  auto indptr_c = indptr.data();
  auto indices_c = indices.data();
  assert(nnodes > required_num * 2); // We cannot sample if there is too few nodes
  std::vector<long> result;
  std::vector<bool> ext(nnodes, false);
  {
    py::gil_scoped_release release;
    long node = -1;
    while (result.size() < required_num) {
      if (node == -1) {
        node = rand() % nnodes;
        if (!ext[node]) {
          ext[node] = true;
          result.push_back(node);
        }
        continue;
      }
      long neighbor_size = indptr_c[node + 1] - indptr_c[node];
      if (neighbor_size == 0) {
        node = -1; // This node has no neighbor, restart random walk
        continue;
      }
      long start = rand() % neighbor_size; // pick a random neighbor
      long base = indptr_c[node];
      for (long i = start;;) {
        long next_node = indices_c[base + i];
        if (!ext[next_node]) { // add to the result
          ext[next_node] = true;
          result.push_back(next_node);
          node = next_node;
          break;
        }
        i = (i + 1) % neighbor_size;
        if (i == start) {
          node = -1; // no neighbor available, restart random walk
          break;
        }
      }
    }
  }
  return V2A(result);
}

class NodeMap {
private:
  std::vector<long> v_;
  int nrank_;
  inline size_t node2key(long node_id, long node_from) {
    return node_id * nrank_ + node_from;
  }
public:
  NodeMap(int nrank):nrank_(nrank) {}
  size_t cnt = 0;
  void insert(long node_id, long node_from) {
    size_t key = node2key(node_id, node_from);
    if (key >= v_.size()) {
      v_.resize(key + 1, -1);
    }
    v_[key] = cnt++;
  }
  long lookup(long node_id, long node_from) {
    size_t key = node2key(node_id, node_from);
    return key < v_.size() ? v_[key] : -1;
  }
};

PyGraph constructGraph(py::list sample_data, PyGraph &graph,
  py::array_t<long> indptr_py, py::array_t<long> indices_py, py::array_t<long> nodes_from_py,
  int rank, int nrank) {
  auto rounds = sample_data.size();
  std::vector<const float *> x;
  std::vector<const int *> y;
  std::vector<const long *> deg, e_id, e_from, heads_id, heads_from;
  std::vector<size_t> num_nodes;

  std::vector<float> graph_x;
  std::vector<int> graph_y;
  std::vector<long> edge_index_u, edge_index_v;
  std::vector<int> fsize;
  //Convert python object to C pointer
  auto indptr = indptr_py.data();
  auto indices = indices_py.data();
  auto nodes_from = nodes_from_py.data();
  for (auto iter=sample_data.begin(); iter!=sample_data.end(); iter++) {
    py::tuple data = iter->cast<py::tuple>();
    auto x_py = data[0].cast<py::array_t<float>>();
    auto y_py = data[1].cast<py::array_t<int>>();
    auto deg_py = data[2].cast<py::array_t<long>>();
    auto edges_py = data[3].cast<py::array_t<long>>();
    auto heads_id_py = data[4].cast<py::array_t<long>>();
    auto heads_from_py = data[5].cast<py::array_t<long>>();
    x.push_back(x_py.data());
    y.push_back(y_py.data());
    deg.push_back(deg_py.data());
    e_id.push_back(edges_py.data());
    e_from.push_back(edges_py.data(1));
    heads_id.push_back(heads_id_py.data());
    heads_from.push_back(heads_from_py.data());
    num_nodes.push_back(heads_id_py.size());
    fsize.push_back(y_py.size());
  }
  {
    py::gil_scoped_release release;
    NodeMap m(nrank);
    // Reindex all the sampled nodes
    for (size_t r = 0; r < rounds; r++) {
      auto n = num_nodes[r];
      for (size_t i = 0; i < n; i++) {
        if (m.lookup(heads_id[r][i], heads_from[r][i]) == -1) {
          m.insert(heads_id[r][i], heads_from[r][i]);
        } else {
          continue;
        }
      }
    }
    graph_x.resize(m.cnt * graph.nFeatures());
    graph_y.resize(m.cnt);
    long current_node_id = 0;
    //Now construct the components for a graph
    for (size_t r = 0; r < rounds; r++) {
      long foreign_num = -1;
      long edges_start = 0;
      auto n = num_nodes[r];
      for (size_t i = 0; i < n; i++) {
        auto id = heads_id[r][i], from = heads_from[r][i];
        if (from != rank) {
          if (foreign_num >= 0) edges_start+=deg[r][foreign_num];
          foreign_num++;
        }
        if (m.lookup(id, from) != current_node_id) continue;
        if (from == rank) {
          //local nodes, get data from graph param
          graph_y[current_node_id] = graph.y_[id];
          std::copy(
            &graph.x_[id*graph.nFeatures()],
            &graph.x_[(id+1)*graph.nFeatures()],
            &graph_x[current_node_id*graph.nFeatures()]
          );
          for (long j = indptr[id]; j<indptr[id+1];j++) {
            long iter = m.lookup(indices[j], nodes_from[j]);
            if (iter == -1) continue;
            edge_index_u.push_back(current_node_id);
            edge_index_v.push_back(iter);
          }
        } else {
          //foreign nodes, get data from sampled data
          graph_y[current_node_id] = y[r][foreign_num];
          std::copy(
            &x[r][foreign_num*graph.nFeatures()],
            &x[r][(foreign_num+1)*graph.nFeatures()],
            &graph_x[current_node_id*graph.nFeatures()]
          );
          for (long j = edges_start; j < edges_start + deg[r][foreign_num]; j++) {
            long iter = m.lookup(e_id[r][j], e_from[r][j]);
            if (iter == -1) continue;
            edge_index_u.push_back(current_node_id);
            edge_index_v.push_back(iter);
          }
        }
        current_node_id++;
      }
      assert(fsize[r] == foreign_num + 1);
    }
    assert(current_node_id == (long)m.cnt);
  }

  return PyGraph(
    std::move(graph_x),
    std::move(graph_y),
    std::move(edge_index_u),
    std::move(edge_index_v),
    graph.nClasses()
  );
}