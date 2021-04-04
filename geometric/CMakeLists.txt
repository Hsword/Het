find_package(pybind11 CONFIG)
find_package(METIS)

if (NOT pybind11_FOUND)
    message(FATAL "pybind11 not found, use `conda install -c conda-forge pybind11`")
endif()

if (NOT METIS_FOUND)
    message(FATAL "METIS target not found, use `conda install -c conda-forge metis`")
endif()

aux_source_directory(csrc SRC_LIST)
pybind11_add_module(libc_GNN ${SRC_LIST})
target_link_libraries(libc_GNN PUBLIC ${METIS_LIBRARIES})
set_property(TARGET libc_GNN PROPERTY CXX_STANDARD 14)
add_custom_target(geometric DEPENDS libc_GNN)
