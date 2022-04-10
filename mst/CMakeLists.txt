# CMakeList.txt : CMake project for mst, include source and define
# project specific logic here.
#
cmake_minimum_required (VERSION 3.8)

project ("mst")

# Add source to this project's executable.
add_executable (mst "mst.cpp" "mst.h" "graph/graph.h" "graph/graph.cpp")

# TODO: Add tests and install targets if needed.
