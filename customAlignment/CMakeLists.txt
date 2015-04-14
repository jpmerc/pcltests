cmake_minimum_required(VERSION 2.8 FATAL_ERROR)

project(customAlignment)

find_package(PCL 1.7 EXACT REQUIRED)
find_package( OpenCV REQUIRED )

include_directories(${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})

add_executable (customAlignment customAlignment.cpp)
target_link_libraries (customAlignment ${PCL_LIBRARIES})

add_executable (customSegmentation customSegmentation.cpp)
target_link_libraries (customSegmentation ${PCL_LIBRARIES} ${OpenCV_LIBS})

add_executable (wholeLoop wholeLoop.cpp)
target_link_libraries (wholeLoop ${PCL_LIBRARIES})

add_executable (wholeLoop_workingExample wholeLoop_workingExample.cpp)
target_link_libraries (wholeLoop_workingExample ${PCL_LIBRARIES})

add_executable (demo demo.cpp)
target_link_libraries (demo ${PCL_LIBRARIES})

add_executable (segmentationAbdeslam segmentationAbdeslam.cpp)
target_link_libraries (segmentationAbdeslam ${PCL_LIBRARIES} ${OpenCV_LIBS})

