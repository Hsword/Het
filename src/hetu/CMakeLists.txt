aux_source_directory(src HETU_SRC_LIST)

find_package(pybind11 CONFIG)

if (NOT pybind11_FOUND)
    message(STATUS "pybind11 not found")
else()
    pybind11_add_module(hetu ${HETU_SRC_LIST})
    target_include_directories(hetu PUBLIC include)
    target_link_libraries(hetu PUBLIC ps)
endif()
