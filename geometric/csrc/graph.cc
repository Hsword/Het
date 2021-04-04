#include "graph.h"

PyGraph makeGraph(py::array_t<float> x, py::array_t<int> y, py::array_t<long> edge_index, int num_classes) {
  assert(x.ndim() == 2);
  assert(y.ndim() == 1);
  assert(edge_index.ndim() == 2 && edge_index.shape(0) == 2);
  auto c_x = A2V(x);
  auto c_y = A2V(y);
  size_t num_edges = edge_index.shape(1);
  std::vector<long> edge_index_u(num_edges), edge_index_v(num_edges);
  memcpy(edge_index_u.data(), edge_index.data(), num_edges * sizeof(long));
  memcpy(edge_index_v.data(), edge_index.data(1), num_edges * sizeof(long));
  return PyGraph(std::move(c_x), std::move(c_y), std::move(edge_index_u), std::move(edge_index_v), num_classes);
}

PyGraph::PyGraph(std::vector<float> &&x, std::vector<int> &&y, std::vector<long> &&edge_index_u,
      std::vector<long> &&edge_index_v, int num_classes) {
  this->x_ = std::move(x);
  this->y_ = std::move(y);
  this->edge_index_u_ = std::move(edge_index_u);
  this->edge_index_v_ = std::move(edge_index_v);
  this->num_classes_ = num_classes;
}

PyGraph::PyGraph(PyGraph &&other) {
  this->x_ = std::move(other.x_);
  this->y_ = std::move(other.y_);
  this->edge_index_u_ = std::move(other.edge_index_u_);
  this->edge_index_v_ = std::move(other.edge_index_v_);
  this->num_classes_ = other.num_classes_;
}

void PyGraph::addSelfLoop() {
  std::vector<bool> check(nNodes(), false);
  for (size_t i = 0;i < nEdges(); i++) {
    if (edge_index_u_[i] == edge_index_v_[i]) {
      check[edge_index_u_[i]] = true;
    }
  }
  for (size_t i = 0;i < nNodes(); i++) {
    if (!check[i]) {
      edge_index_u_.push_back(i);
      edge_index_v_.push_back(i);
    }
  }
}

void PyGraph::removeSelfLoop() {
  std::vector<long> u, v;
  for (size_t i = 0;i < nEdges(); i++) {
    if (edge_index_u_[i] != edge_index_v_[i]) {
      u.push_back(edge_index_u_[i]);
      v.push_back(edge_index_v_[i]);
    }
  }
  edge_index_u_ = std::move(u);
  edge_index_v_ = std::move(v);
}

double PyGraph::denseEfficiency() {
  return double(nEdges()) / (nNodes() * nNodes());
}

std::vector<long> PyGraph::degree() {
  std::vector<long> deg(nNodes(), 0);
  for (size_t i = 0;i < nEdges(); i++) {
    deg[edge_index_u_[i]]++;
  }
  return deg;
}

py::array_t<double> PyGraph::gcnNorm(bool use_original_gcn_norm) {
  auto deg = degree();
  std::vector<double> norm(nEdges());
  if (use_original_gcn_norm) {
    for (size_t i = 0;i < nEdges(); i++) {
      long v = edge_index_v_[i], u = edge_index_u_[i];
      norm[i] = sqrt(1.0f / (deg[v] * deg[u]));
    }
  } else {
    for (size_t i = 0;i < nEdges(); i++) {
      long v = edge_index_v_[i];
      norm[i] = 1.0f / deg[v];
    }
  }
  return V2A(norm);
}

std::vector<idx_t> PyGraph::partition(idx_t nparts, bool balance_edge) {
  assert(nparts >= 1);
  if (nparts == 1) {
    return std::vector<idx_t>(nNodes(), 0);
  }
  std::vector<idx_t> indices(nEdges()), indptr(nNodes() + 1);
  auto deg = degree();
  indptr[0] = 0;
  for (size_t i = 1; i <= nNodes(); i++) {
    indptr[i] = deg[i-1] + indptr[i-1];
  }
  auto temp = indptr;
  for (size_t i = 0; i < nEdges(); i++) {
    long u = edge_index_u_[i], v = edge_index_v_[i];
    indices[temp[u]] = v;
    temp[u]++;
  }
  //Start metis API
  idx_t num_nodes = nNodes();
  idx_t ncon = 1, edgecut;
  std::vector<idx_t> parts(nNodes());

  auto partition_function = METIS_PartGraphKway;
  if (nparts > 8) {
    partition_function = METIS_PartGraphRecursive;
  }
  // Decide whether to balance edge
  // two constraint, number of nodes and nodes' degree
  std::vector<idx_t> vwgt;
  idx_t *vwgt_data = NULL;
  if (balance_edge) {
    ncon = 2;
    vwgt.resize(2 * nNodes());
    for (size_t i = 0;i < nNodes(); i++) {
      vwgt[i * ncon] = deg[i];
      vwgt[i * ncon + 1] = 1;
    }
    vwgt_data = vwgt.data();
  }
  int info = partition_function(
    &num_nodes, /* number of nodes */
    &ncon,
    indptr.data(),
    indices.data(),
    vwgt_data,    /* weight of nodes */
    NULL,    /* The size of the vertices for computing the total communication volume */
    NULL,    /* weight of edges */
    &nparts, /* num parts */
    NULL,    /* the desired weight for each partition and constraint */
    NULL,    /* an array of size ncon that specifies the allowed load imbalance tolerance for each constraint */
    NULL,    /* options */
    &edgecut,  /* store number of edge cut */
    parts.data() /* store partition result */
  );
  switch (info) {
  case METIS_OK:
    break;
  case METIS_ERROR_INPUT:
    printf("Metis error input");
    break;
  case METIS_ERROR_MEMORY:
    printf("Metis error memory");
    break;
  case METIS_ERROR:
  default:
    printf("Metis error");
    break;
  }
  assert(info == METIS_OK);
  return parts;
}

py::array_t<idx_t> PyGraph::PyPartition(idx_t nparts) {
  auto x = partition(nparts, true);
  return V2A(x);
}

py::tuple PyGraph::part_graph(int nparts, bool balance_edge) {
  auto nodes = partition((idx_t)nparts, balance_edge);
  std::vector<long> reindex(nNodes()), counting(nparts, 0);
  for (size_t i = 0;i < nNodes(); i++) {
    reindex[i] = counting[nodes[i]];
    counting[nodes[i]]++;
  }

  // put edges in nparts * nparts bucket
  std::vector<std::vector <long>> edges(nparts * nparts);
  std::vector<std::vector <long>> edges_u(nparts), edges_v(nparts);
  // compute reindexed edges
  std::vector<long> reindex_u(nEdges()), reindex_v(nEdges());
  for (size_t i = 0;i < nEdges(); i++) {
    auto u = edge_index_u_[i], v = edge_index_v_[i];
    reindex_u[i] = reindex[u];
    reindex_v[i] = reindex[v];

    int where = nparts * nodes[u] + nodes[v];
    edges[where].push_back(i);
    if (nodes[u] == nodes[v]) {
      edges_u[nodes[u]].push_back(reindex[u]);
      edges_v[nodes[v]].push_back(reindex[v]);
    }
  }

  // construct subgraphs
  py::list subgraphs;
  for (int part = 0; part < nparts; part++) {
    std::vector<float> x(nFeatures() * counting[part]);
    std::vector<int> y(counting[part]);
    for (size_t i = 0; i < nNodes(); i++) {
      if (nodes[i] == part) {
        y[reindex[i]] = y_[i];
        memcpy(&x[reindex[i]*nFeatures()], &x_[i*nFeatures()], sizeof(float)*nFeatures());
      }
    }
    subgraphs.append(std::make_shared<PyGraph>(
      std::move(x),
      std::move(y),
      std::move(edges_u[part]),
      std::move(edges_v[part]),
      nClasses()
    ));
  }
  //Transform edges-list into python-list
  py::list py_edge_list;
  for (int i = 0; i < nparts; i++) {
    py::list temp_list;
    for (int j = 0; j < nparts; j++) {
      temp_list.append(V2A(edges[i * nparts + j]));
    }
    py_edge_list.append(temp_list);
  }
  return py::make_tuple(
    subgraphs,
    py_edge_list,
    py::make_tuple(V2A(reindex_u), V2A(reindex_v))
  );
}
