cmake_minimum_required(VERSION 2.8 FATAL_ERROR)

project(lccp)

find_package(PCL 1.8 REQUIRED)
find_package( OpenCV REQUIRED )

include_directories(${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})

add_executable (lccp lccp.cpp)
target_link_libraries (lccp ${PCL_LIBRARIES})

add_executable (mean_lccp mean_lccp.cpp)
target_link_libraries (mean_lccp ${PCL_LIBRARIES} ${OpenCV_LIBS})
