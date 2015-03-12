cmake_minimum_required(VERSION 2.8 FATAL_ERROR)

project(normalDistribution)

find_package(PCL 1.7 REQUIRED)

include_directories(${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})

add_executable (normalDistribution normalDistribution.cpp)
target_link_libraries (normalDistribution ${PCL_LIBRARIES})


