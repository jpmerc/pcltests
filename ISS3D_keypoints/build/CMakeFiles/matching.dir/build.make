# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 2.8

#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:

# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list

# Suppress display of executed commands.
$(VERBOSE).SILENT:

# A target that is always out of date.
cmake_force:
.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E remove -f

# The program to use to edit the cache.
CMAKE_EDIT_COMMAND = /usr/bin/ccmake

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/jp/pcltests/ISS3D_keypoints

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/jp/pcltests/ISS3D_keypoints/build

# Include any dependencies generated for this target.
include CMakeFiles/matching.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/matching.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/matching.dir/flags.make

CMakeFiles/matching.dir/matching.cpp.o: CMakeFiles/matching.dir/flags.make
CMakeFiles/matching.dir/matching.cpp.o: ../matching.cpp
	$(CMAKE_COMMAND) -E cmake_progress_report /home/jp/pcltests/ISS3D_keypoints/build/CMakeFiles $(CMAKE_PROGRESS_1)
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Building CXX object CMakeFiles/matching.dir/matching.cpp.o"
	/usr/bin/c++   $(CXX_DEFINES) $(CXX_FLAGS) -o CMakeFiles/matching.dir/matching.cpp.o -c /home/jp/pcltests/ISS3D_keypoints/matching.cpp

CMakeFiles/matching.dir/matching.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/matching.dir/matching.cpp.i"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -E /home/jp/pcltests/ISS3D_keypoints/matching.cpp > CMakeFiles/matching.dir/matching.cpp.i

CMakeFiles/matching.dir/matching.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/matching.dir/matching.cpp.s"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -S /home/jp/pcltests/ISS3D_keypoints/matching.cpp -o CMakeFiles/matching.dir/matching.cpp.s

CMakeFiles/matching.dir/matching.cpp.o.requires:
.PHONY : CMakeFiles/matching.dir/matching.cpp.o.requires

CMakeFiles/matching.dir/matching.cpp.o.provides: CMakeFiles/matching.dir/matching.cpp.o.requires
	$(MAKE) -f CMakeFiles/matching.dir/build.make CMakeFiles/matching.dir/matching.cpp.o.provides.build
.PHONY : CMakeFiles/matching.dir/matching.cpp.o.provides

CMakeFiles/matching.dir/matching.cpp.o.provides.build: CMakeFiles/matching.dir/matching.cpp.o

# Object files for target matching
matching_OBJECTS = \
"CMakeFiles/matching.dir/matching.cpp.o"

# External object files for target matching
matching_EXTERNAL_OBJECTS =

matching: CMakeFiles/matching.dir/matching.cpp.o
matching: /usr/lib/libboost_system-mt.so
matching: /usr/lib/libboost_filesystem-mt.so
matching: /usr/lib/libboost_thread-mt.so
matching: /usr/lib/libboost_date_time-mt.so
matching: /usr/lib/libboost_iostreams-mt.so
matching: /usr/lib/libboost_serialization-mt.so
matching: /usr/lib/libpcl_common.so
matching: /usr/lib/libflann_cpp_s.a
matching: /usr/lib/libpcl_kdtree.so
matching: /usr/lib/libpcl_octree.so
matching: /usr/lib/libpcl_search.so
matching: /usr/lib/libOpenNI.so
matching: /usr/lib/libvtkCommon.so.5.8.0
matching: /usr/lib/libvtkRendering.so.5.8.0
matching: /usr/lib/libvtkHybrid.so.5.8.0
matching: /usr/lib/libvtkCharts.so.5.8.0
matching: /usr/lib/libpcl_io.so
matching: /usr/lib/libpcl_sample_consensus.so
matching: /usr/lib/libpcl_filters.so
matching: /usr/lib/libpcl_visualization.so
matching: /usr/lib/libpcl_outofcore.so
matching: /usr/lib/libpcl_features.so
matching: /usr/lib/libpcl_segmentation.so
matching: /usr/lib/libpcl_people.so
matching: /usr/lib/libpcl_registration.so
matching: /usr/lib/libpcl_recognition.so
matching: /usr/lib/libpcl_keypoints.so
matching: /usr/lib/libqhull.so
matching: /usr/lib/libpcl_surface.so
matching: /usr/lib/libpcl_tracking.so
matching: /usr/lib/libpcl_apps.so
matching: /usr/lib/libboost_system-mt.so
matching: /usr/lib/libboost_filesystem-mt.so
matching: /usr/lib/libboost_thread-mt.so
matching: /usr/lib/libboost_date_time-mt.so
matching: /usr/lib/libboost_iostreams-mt.so
matching: /usr/lib/libboost_serialization-mt.so
matching: /usr/lib/libqhull.so
matching: /usr/lib/libOpenNI.so
matching: /usr/lib/libflann_cpp_s.a
matching: /usr/lib/libvtkCommon.so.5.8.0
matching: /usr/lib/libvtkRendering.so.5.8.0
matching: /usr/lib/libvtkHybrid.so.5.8.0
matching: /usr/lib/libvtkCharts.so.5.8.0
matching: /usr/lib/libpcl_common.so
matching: /usr/lib/libpcl_kdtree.so
matching: /usr/lib/libpcl_octree.so
matching: /usr/lib/libpcl_search.so
matching: /usr/lib/libpcl_io.so
matching: /usr/lib/libpcl_sample_consensus.so
matching: /usr/lib/libpcl_filters.so
matching: /usr/lib/libpcl_visualization.so
matching: /usr/lib/libpcl_outofcore.so
matching: /usr/lib/libpcl_features.so
matching: /usr/lib/libpcl_segmentation.so
matching: /usr/lib/libpcl_people.so
matching: /usr/lib/libpcl_registration.so
matching: /usr/lib/libpcl_recognition.so
matching: /usr/lib/libpcl_keypoints.so
matching: /usr/lib/libpcl_surface.so
matching: /usr/lib/libpcl_tracking.so
matching: /usr/lib/libpcl_apps.so
matching: /usr/lib/libvtkViews.so.5.8.0
matching: /usr/lib/libvtkInfovis.so.5.8.0
matching: /usr/lib/libvtkWidgets.so.5.8.0
matching: /usr/lib/libvtkHybrid.so.5.8.0
matching: /usr/lib/libvtkParallel.so.5.8.0
matching: /usr/lib/libvtkVolumeRendering.so.5.8.0
matching: /usr/lib/libvtkRendering.so.5.8.0
matching: /usr/lib/libvtkGraphics.so.5.8.0
matching: /usr/lib/libvtkImaging.so.5.8.0
matching: /usr/lib/libvtkIO.so.5.8.0
matching: /usr/lib/libvtkFiltering.so.5.8.0
matching: /usr/lib/libvtkCommon.so.5.8.0
matching: /usr/lib/libvtksys.so.5.8.0
matching: CMakeFiles/matching.dir/build.make
matching: CMakeFiles/matching.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --red --bold "Linking CXX executable matching"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/matching.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/matching.dir/build: matching
.PHONY : CMakeFiles/matching.dir/build

CMakeFiles/matching.dir/requires: CMakeFiles/matching.dir/matching.cpp.o.requires
.PHONY : CMakeFiles/matching.dir/requires

CMakeFiles/matching.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/matching.dir/cmake_clean.cmake
.PHONY : CMakeFiles/matching.dir/clean

CMakeFiles/matching.dir/depend:
	cd /home/jp/pcltests/ISS3D_keypoints/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/jp/pcltests/ISS3D_keypoints /home/jp/pcltests/ISS3D_keypoints /home/jp/pcltests/ISS3D_keypoints/build /home/jp/pcltests/ISS3D_keypoints/build /home/jp/pcltests/ISS3D_keypoints/build/CMakeFiles/matching.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/matching.dir/depend

