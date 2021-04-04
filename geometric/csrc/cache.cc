#include "cache.h"

void PyCache::insertItem(long node_id, long node_from, py::array_t<float> x, int y,
    py::array_t<long> edges_id, py::array_t<long> edges_from, double priority) {
  std::lock_guard<std::mutex> l(mtx);
  this->insert({node_id, node_from, priority}, {A2V(x), y, A2V(edges_id), A2V(edges_from)});
}

py::object PyCache::queryItem(long node_id, long node_from) {
  std::lock_guard<std::mutex> l(mtx);
  if (this->hasKey({node_id, node_from})) {
    auto v = this->lookup({node_id, node_from});
    return py::make_tuple(V2A(v.x), v.y, V2A(v.edges_id), V2A(v.edges_from));
  }
  return py::none();
}

py::tuple PyCache::getKeys() {
  std::lock_guard<std::mutex> l(mtx);
  std::vector<long> a, b;
  for (auto iter = map_.begin(); iter != map_.end(); iter++) {
    a.push_back(iter->first.node_id);
    b.push_back(iter->first.node_from);
  }
  return py::make_tuple(V2A(a) , V2A(b));
}

py::tuple PyCache::queryItemPacked(py::array_t<long> node_ids, py::array_t<long> node_froms) {
  size_t n = node_ids.size();
  auto node_ids_ = node_ids.data();
  auto node_froms_ = node_froms.data();
  std::vector<size_t> idx_hit, idx_miss;
  std::vector<int> y;
  std::vector<float> x;
  std::vector<long> deg, eid, efrom;
  {
    py::gil_scoped_release release;
    std::vector<NodeData> vd;
    size_t e_size = 0, x_size = 0;
    for (size_t i = 0; i < n; i++) {
      std::lock_guard<std::mutex> l(mtx);
      if (this->hasKey({node_ids_[i], node_froms_[i]})) {
        idx_hit.push_back(i);
        auto v = this->lookup({node_ids_[i], node_froms_[i]});
        vd.push_back(v);
        e_size += v.edges_from.size();
        x_size += v.x.size();
      } else {
        idx_miss.push_back(i);
      }
    }
    x.resize(x_size);
    eid.resize(e_size);
    efrom.resize(e_size);
    size_t x_pos = 0, e_pos = 0;
    for (size_t i = 0; i < vd.size(); i++) {
      auto &v = vd[i];
      std::copy(v.x.begin(), v.x.end(), &x[x_pos]);
      y.push_back(v.y);
      deg.push_back(v.edges_from.size());
      std::copy(v.edges_id.begin(), v.edges_id.end(), &eid[e_pos]);
      std::copy(v.edges_from.begin(), v.edges_from.end(), &efrom[e_pos]);
      e_pos += v.edges_from.size();
      x_pos += v.x.size();
    }
  }
  return py::make_tuple(V2A(idx_hit), V2A(idx_miss), V2A(x), V2A(y), V2A(deg), V2A(eid), V2A(efrom));
}


PyCache* makeCache(size_t limit, std::string policy) {
  CachePolicy<Node> *cp;
  if (policy == "FIFO")
    cp = new FIFOCachePolicy<Node>();
  else if (policy == "Priority")
    cp = new PriorityCachePolicy<Node>();
  else {
    throw std::runtime_error("Unknown Cache Policy");
  }
  auto obj = new PyCache(limit, cp);
  return obj;
}