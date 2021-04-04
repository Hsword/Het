#pragma once

#include "PSFunc.h"

namespace ps {

enum InitType {
  Constant,
  Uniform,
  Normal,
  TruncatedNormal,
};

template<> struct PSFData<ParamInit> {
  using Request = tuple<
    Key, // key
    int, // param_type
    size_t, // len
    size_t, // width
    int, // init_type
    double, // init_a
    double // init_b
  >;
  using Response = tuple<>;
  static void _callback(const Response &response) {}
};

template<> struct PSFData<ParamClear> {
  using Request = tuple<
    Key // key
  >;
  using Response = tuple<>;
  static void _callback(const Response &response) {}
};

template<> struct PSFData<ParamSave> {
  using Request = tuple<
    Key,
    SArray<char>, // address
    bool // different from load
  >;
  using Response = tuple<>;
  static void _callback(const Response &response) {}
};

template<> struct PSFData<ParamLoad> {
  using Request = tuple<
    Key,
    SArray<char> // address
  >;
  using Response = tuple<>;
  static void _callback(const Response &response) {}
};

} // namespace ps
