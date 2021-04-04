#ifndef GNN_COMMON_H
#define GNN_COMMON_H

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include <vector>
#include <iostream>

namespace py = pybind11;

template <typename T>
py::array_t<T> V2A(std::vector<T> &v) {
  return py::array_t<T>(v.size(), v.data());
}

template <typename T>
py::array_t<T> V2ANOCOPY(std::vector<T> &v) {
  return py::array_t<T>(v.size(), v.data(), py::none());
}

template <typename T>
std::vector<T> A2V(py::array_t<T> &arr) {
  std::vector<T> v(arr.size());
  memcpy(v.data(), arr.data(), v.size() * sizeof(T));
  return v;
}

#endif /* GNN_COMMON_H */