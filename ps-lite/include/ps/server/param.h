#pragma once

#include <vector>

#include "common/shared_mutex.h"
#include "ps/psf/PSFunc.h"

namespace ps {

enum ParamType {
  kParam,
  kParam2D,
  kCacheTable,
};

/*
  Param with a read-write lock
*/
template<typename V>
class Param{
public:
    explicit Param(size_t size) {
        vec_ = new V[size]();
        size_ = size;
    }

    ~Param() {
        delete [] vec_;
    }

    Param(const Param &) = delete;

    s_lock<4> read_guard() const noexcept {
        return s_lock<4>(mtx);
    }
    x_lock<4> write_guard() noexcept {
        return x_lock<4>(mtx);
    }

    inline const V* data() const { return vec_; }
    inline V* data() { return vec_; }
    inline V* begin() { return data(); }
    inline V* end() { return data() + size(); }
    inline V& operator[] (size_t i) { return vec_[i]; }
    inline const V& operator[] (size_t i) const { return vec_[i]; }
    inline size_t size() const { return size_; }
    virtual constexpr ParamType type() { return kParam; }
private:
    mutable shared_mutex<4> mtx;
    V* vec_;
    size_t size_;
};


template<typename V>
class Param2D : public Param<V> {
public:
    explicit Param2D(size_t len, size_t wid): Param<V>(len * wid) {
        length = len;
        width  = wid;
    }
    constexpr ParamType type() { return kParam2D; }
    size_t length, width;
};

template<typename V>
class CacheTable : public Param2D<V> {
public:
    explicit CacheTable(size_t len, size_t wid) : Param2D<V>(len, wid) {
        ver = new version_t[len]();
    }
    ~CacheTable() {
        delete [] ver;
    }
    constexpr ParamType type() { return kCacheTable; }
    version_t* ver;
};


} // namespace ps
