#ifndef GNN_CACHE_H
#define GNN_CACHE_H

#include "common.h"

#include <map>
#include <queue>
#include <mutex>

// Virtual Base class of CachePolicy
template <typename Key>
class CachePolicy {
public:
  //Called when a new Key is inserted into the cache
  virtual void onInsert(Key k) = 0;
  //Called before insert, when the cache is full
  virtual Key selectRemove(Key k) = 0;
  virtual ~CachePolicy() {}
};

/*
  Cache: the base cache template
    Use mutex to ensure that calls to it is thread-safe
    provide: hasKey(Key), lookup(Key), insert(Key)
*/
template <typename Key, typename Data>
class Cache{
protected:
  std::map<Key, Data> map_;
  size_t item_limit_;
  std::shared_ptr<CachePolicy<Key>> cp_;
  bool _hasKey(Key k) {
    return map_.find(k) != map_.end();
  }
public:
	Cache(size_t item_limit, CachePolicy<Key> *cp) : item_limit_(item_limit), cp_(cp) {};
	~Cache() {}
  bool hasKey(Key k) {
    return _hasKey(k);
  }
  Data& lookup(Key k) {
    return map_.at(k);
  }
  void insert(Key k, Data d) {
    if (_hasKey(k)) {
      return;
    }
    if (map_.size() == item_limit_) {
      Key evict_key = cp_->selectRemove(k);
      map_.erase(evict_key);
    }
    map_[k] = d;
    cp_->onInsert(k);
  }
  //getMap : used for debug only
  auto getMap() {return map_;}
};

//Definition for out Key and Value
struct Node {
  long node_id;
  long node_from;
  //Some cache policy may use this
  double priority;

  //Required for std::map
  bool operator<(const Node &other) const {
    return node_from == other.node_from ?
      node_id < other.node_id : node_from < other.node_from;
  }

  Node(long node_id, long node_from, double priority=0) :
    node_id(node_id), node_from(node_from), priority(priority) {}
};

struct NodeData {
  std::vector<float> x;
  int y;
  std::vector<long> edges_id, edges_from;
};

//Wrapper for Cache in python
class PyCache : Cache<Node, NodeData> {
private:
  std::mutex mtx;
public:
  using Cache<Node, NodeData>::Cache;

  py::object queryItem(long node_id, long node_from);
  py::tuple queryItemPacked(py::array_t<long> node_ids, py::array_t<long> node_froms);
  void insertItem(long node_id, long node_from, py::array_t<float> x, int y,
    py::array_t<long> edges_id, py::array_t<long> edges_from,double priority = 0);

  //used for debugging
  py::tuple getKeys();
  auto getLimit() {return item_limit_;}
  auto getSize() {return map_.size();}
};

// Factory function
PyCache* makeCache(size_t limit, std::string policy);

//-------------------------------------------------------------------------------------
//Below implement some cache policy here

template<typename Key>
class FIFOCachePolicy: public CachePolicy<Key> {
private:
  std::queue<Key> queue_;
  size_t limit_;
public:
  void onInsert(Key k) {
    queue_.push(k);
  }

  Key selectRemove(Key k) {
    auto remove_key = queue_.front();
    queue_.pop();
    return remove_key;
  }

  ~FIFOCachePolicy() {}
};

template<typename Key>
class PriorityCachePolicy: public CachePolicy<Key> {
private:
  struct cmp
  {
    bool operator()(Key a,Key b)
    {
      return a.priority > b.priority;
    }
  };
  std::priority_queue<Key, std::vector<Key>, cmp> queue_;
  size_t limit_;
public:
  void onInsert(Key k) {
    queue_.push(k);
  }

  Key selectRemove(Key k) {
    auto remove_key = queue_.top();
    queue_.pop();
    return remove_key;
  }

  ~PriorityCachePolicy() {}
};

#endif /* GNN_CACHE_H */