cmake_minimum_required(VERSION 2.8 FATAL_ERROR)

project(multipleKinects)

find_package(PCL 1.7 REQUIRED)

include_directories(${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})

add_executable (multipleKinects multipleKinects.cpp)
target_link_libraries (multipleKinects ${PCL_LIBRARIES})

add_executable (containerAlignment containerAlignment.cpp)
target_link_libraries (containerAlignment ${PCL_LIBRARIES})
