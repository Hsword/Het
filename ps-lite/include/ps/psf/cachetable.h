#pragma once

#include "PSFunc.h"

namespace ps {

typedef int64_t version_t;

template<> struct PSFData<kPushEmbedding> {
  using Request = tuple<
    Key, // key
    SArray<size_t>, // rows
    SArray<float>, // data
    SArray<version_t> // updates
  >;
  using Response = tuple<>;
  static void _callback(const Response &response) {}
};

template<> struct PSFData<kSyncEmbedding> {
  using Request = tuple<
    Key, // key
    SArray<size_t>, // rows
    SArray<version_t>, // current version
    version_t // bound
  >;
  using Response = tuple<
    SArray<size_t>, // rows that should be updated
    SArray<version_t>, // server version returned
    SArray<float> // embedding value
  >;
  // Use a closure to pass cached embedding data target
  typedef std::function<void(const Response &response, size_t offset)> Closure;
};

template<> struct PSFData<kPushSyncEmbedding> {
  using Request = tuple<
    Key, // key
    SArray<size_t>, // rows
    SArray<version_t>, // current version
    version_t, // bound
    SArray<size_t>, // push rows
    SArray<float>, // push data
    SArray<version_t> // push updates
  >;
  using Response = PSFData<kSyncEmbedding>::Response;
};

} // namespace ps
