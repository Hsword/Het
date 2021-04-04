#include "sampler.h"

DistributedSampler::DistributedSampler(DistributedSampler &&other) {
  this->indices_ = std::move(other.indices_);
  this->indptr_ = std::move(other.indptr_);
  this->nodes_from_ = std::move(other.nodes_from_);
  this->local_degree_ = std::move(other.local_degree_);
  this->rank_ = other.rank_;
}

DistributedSampler::DistributedSampler(std::vector<long> &&indptr, std::vector<long> &&indices,
                                      std::vector<long> &&nodes_from, int rank) {
  this->indices_ = std::move(indices);
  this->indptr_ = std::move(indptr);
  this->nodes_from_ = std::move(nodes_from);
  this->local_degree_ = std::vector<long>(this->indptr_.size() - 1);
  this->rank_ = rank;
  this->sortIndices();
}

static size_t _sortIndices(long *st, long *st_r, const size_t len, const long rank) {
  if (len == 0) return 0;
  size_t l = 0, r = len - 1;
  while(l < r) {
    if (st_r[l] == rank) {
      // local edge
      l++;
    } else {
      // remote edge
      std::swap(st_r[l], st_r[r]);
      std::swap(st[l], st[r]);
      r--;
    }
  }
  return l;
}

void DistributedSampler::sortIndices() {
  for (size_t i = 0; i < indptr_.size() - 1; i++) {
    size_t len = indptr_[i + 1] - indptr_[i];
    size_t idx = indptr_[i];
    auto deg = _sortIndices(&indices_[idx], &nodes_from_[idx], len, rank_);
    this->local_degree_[i] = deg;
  }
}

py::tuple DistributedSampler::sampleNeighbours(py::array_t<long> node_ids, double ratio) {
  size_t num = node_ids.size();
  std::vector <long> neighbor_id(num), neighbor_from(num);
  auto node_ids_ = node_ids.data();
  {
    //Release python lock, C++ code block
    py::gil_scoped_release release;
    for (size_t i = 0; i < num; i++) {
      auto node = node_ids_[i];
      size_t deg = indptr_[node + 1] - indptr_[node];
      if (deg == 0) {
        neighbor_id[i] = node;
        neighbor_from[i] = rank_;
        continue;
      }

      if ((double)random()/RAND_MAX < ratio && local_degree_[node] > 0) {
        auto base = indptr_[node];
        auto offset = random() % local_degree_[node];
        neighbor_id[i] = indices_[base+offset];
        neighbor_from[i] = nodes_from_[base+offset];
      } else {
        auto base = indptr_[node];
        auto offset = random() % deg;
        neighbor_id[i] = indices_[base+offset];
        neighbor_from[i] = nodes_from_[base+offset];
      }
    }
  }
  return py::make_tuple(V2A(neighbor_id), V2A(neighbor_from));
}

py::tuple DistributedSampler::generateLocalGraph(py::array_t<long> node_ids) {
  assert(node_ids.ndim() == 1);
  size_t num_nodes = node_ids.shape(0);
  auto node_ids_ = node_ids.data();
  std::vector<long> u, v; //The return value
  {
    //Release python lock, C++ code block
    py::gil_scoped_release release;
    std::vector<long> reindex(nLocalNodes(), -1);
    for (size_t i = 0; i < num_nodes; i++) {
      auto node = node_ids_[i];
      if (node > (long)nLocalNodes()) {
        throw std::runtime_error("node_ids value error");
      }
      reindex[node] = i;
    }
    for (size_t i = 0; i < num_nodes; i++) {
      auto node = node_ids_[i];
      auto s_i = indptr_[node], e_i = indptr_[node+1];
      for (auto j = s_i; j < e_i; j++) {
        if (reindex[indices_[j]] >= 0) {
          u.push_back(i);
          v.push_back(reindex[indices_[j]]);
        }
      }
    }
  }
  return py::make_tuple(V2A(u), V2A(v));
}

DistributedSampler makeSampler(py::list edges, size_t rank, size_t num_nodes) {
  size_t total = 0;
  std::vector<size_t> deg(num_nodes, 0);
  for(auto iter = edges.begin(); iter != edges.end(); ++iter) {
    auto edge = iter->cast<py::array_t<long>>();
    assert(edge.ndim() == 2 && edge.shape(0) == 2);
    size_t num_edges = edge.shape(1);
    for (size_t i = 0; i < num_edges; i++) {
      deg[edge.at(0, i)]++;
    }
    total += num_edges;
  }
  std::vector<long> indptr(num_nodes+1), indices(total), nodes_from(total);
  indptr[0] = 0;
  for (size_t i = 1; i <= num_nodes; i++) {
    indptr[i] = deg[i-1] + indptr[i-1];
  }
  auto temp = indptr;
  int subgraph_id = 0;
  for(auto iter = edges.begin(); iter != edges.end(); ++iter, ++subgraph_id) {
    auto edge = iter->cast<py::array_t<long>>();
    size_t num_edges = edge.shape(1);
    for (size_t i = 0; i < num_edges; i++) {
      long u = edge.at(0, i), v = edge.at(1, i);
      indices[temp[u]] = v;
      nodes_from[temp[u]] = subgraph_id;
      temp[u]++;
    }
  }

  return DistributedSampler(std::move(indptr), std::move(indices), std::move(nodes_from), rank);
}