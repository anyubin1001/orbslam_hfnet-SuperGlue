cmake_minimum_required(VERSION 2.8.3)
project(omni_vslam)

##===================================just for cmake settings==========================================================##
## Compile as C++11, supported in ROS Kinetic and newer
SET(CMAKE_BUILD_TYPE Release)
MESSAGE("-- Build type: " ${CMAKE_BUILD_TYPE})

#set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS}  -Wall  -O3 -march=native ")
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall   -O3 -march=native")

# Check C++11 or C++0x support
include(CheckCXXCompilerFlag)
CHECK_CXX_COMPILER_FLAG("-std=c++11" COMPILER_SUPPORTS_CXX11)
CHECK_CXX_COMPILER_FLAG("-std=c++0x" COMPILER_SUPPORTS_CXX0X)
if (COMPILER_SUPPORTS_CXX11)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
    add_definitions(-DCOMPILEDWITHC11)
    message(STATUS "Using flag -std=c++11.")
elseif (COMPILER_SUPPORTS_CXX0X)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++0x")
    add_definitions(-DCOMPILEDWITHC0X)
    message(STATUS "Using flag -std=c++0x.")
else ()
    message(FATAL_ERROR "The compiler ${CMAKE_CXX_COMPILER} has no C++11 support. Please use a different C++ compiler.")
endif ()
##===================================just for cmake settings==========================================================##


#open warning
#if ( CMAKE_COMPILER_IS_GNUCC )
#    set(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} -Wall -Wextra")
#endif()
#if ( MSVC )
#    set(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} /W4")
#endif()

set(Torch_DIR ${PROJECT_SOURCE_DIR}/Thirdparty/libtorch/share/cmake/Torch)
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -std=c++11 -W")  # 指定编译器
find_package(OpenCV 3.4.7 REQUIRED)
find_package(Pangolin REQUIRED)
find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

#set(EIGEN3_INCLUDE_DIR ${PROJECT_SOURCE_DIR}/Thirdparty/eigen)



include_directories(
/usr/include/eigen3

${PROJECT_SOURCE_DIR}
${PROJECT_SOURCE_DIR}/include
#${EIGEN3_INCLUDE_DIR}
${Pangolin_INCLUDE_DIRS}
${OpenCV_INCLUDE_DIRS}
${PROJECT_SOURCE_DIR}/Thirdparty/opengv/include

${PROJECT_SOURCE_DIR}/Thirdparty/tensorflow_r1.14/
${PROJECT_SOURCE_DIR}/Thirdparty/tensorflow_r1.14/bazel-genfiles
${PROJECT_SOURCE_DIR}/Thirdparty/tensorflow_r1.14/tensorflow/contrib/makefile/downloads/absl
${PROJECT_SOURCE_DIR}/Thirdparty/abseil-cpp

${PROJECT_SOURCE_DIR}/Thirdparty/tensorflow_r1.14/tensorflow/contrib/resampler/kernels
)

add_subdirectory(${PROJECT_SOURCE_DIR}/Thirdparty/DBoW3)
add_subdirectory(${PROJECT_SOURCE_DIR}/Thirdparty/g2o)
add_subdirectory(${PROJECT_SOURCE_DIR}/Thirdparty/models)
add_subdirectory(${PROJECT_SOURCE_DIR}/Thirdparty/opengv)

add_dependencies(g2o
models
)

set(cv_bridge_DIR /usr/local/opencv3.4.7/cv_bridge/share/cv_bridge/cmake)   #!!!!!
find_package(catkin REQUIRED COMPONENTS
  cv_bridge
  image_transport
  nodelet
  roscpp
  rospy
  sensor_msgs 
  std_msgs
  tf
)

catkin_package(
  #INCLUDE_DIRS include
  LIBRARIES omni_vslam
  CATKIN_DEPENDS cv_bridge image_transport nodelet roscpp rospy sensor_msgs std_msgs
  DEPENDS system_lib
)

set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/lib)
link_directories(${PROJECT_SOURCE_DIR}/Thirdparty/tensorflow_r1.14/bazel-bin/tensorflow)  # 动态链接库目录

add_library(${PROJECT_NAME} SHARED
src/System.cc
src/Tracking.cc
src/LocalMapping.cc
src/LoopClosing.cc
src/Random.cpp
src/Timestamp.cpp
src/ORBextractor.cc
src/ORBmatcher.cc
src/FrameDrawer.cc
src/Converter.cc
src/MapPoint.cc
src/KeyFrame.cc
src/Map.cc
src/MapDrawer.cc
src/Optimizer.cc
src/PnPsolver.cc
src/Frame.cc
src/KeyFrameDatabase.cc
src/Sim3Solver.cc
src/Initializer.cc
src/Viewer.cc
${PROJECT_SOURCE_DIR}/Thirdparty/tensorflow_r1.14/tensorflow/contrib/resampler/kernels/resampler_ops.cc 
${PROJECT_SOURCE_DIR}/Thirdparty/tensorflow_r1.14/tensorflow/contrib/resampler/kernels/resampler_ops_gpu.cu.cc
)

add_dependencies(${PROJECT_NAME}
g2o
DBoW3
opengv
)

###########
## Build ##
###########

## Specify additional locations of header files
## Your package locations should be listed before other locations
include_directories(
# include
  ${catkin_INCLUDE_DIRS}  
)

message(STATUS "Compile With map save/load function")
find_library(BOOST_SERIALIZATION boost_serialization)

if (NOT BOOST_SERIALIZATION)
  message(FATAL_ERROR "Can't find libboost_serialization")
endif()

target_link_libraries(${PROJECT_NAME}
${OpenCV_LIBS}
#${EIGEN3_LIBS}
${Pangolin_LIBRARIES}
${PROJECT_SOURCE_DIR}/lib/libDBoW3.so
${PROJECT_SOURCE_DIR}/lib/libg2o.so
${PROJECT_SOURCE_DIR}/lib/libmodels.so
${PROJECT_SOURCE_DIR}/lib/libopengv.so
${TORCH_LIBRARIES}
/usr/local/lib/libnabo.a
tensorflow_cc tensorflow_framework
)

target_compile_definitions(${PROJECT_NAME} PUBLIC FUNC_MAP_SAVE_LOAD=1)

# build the omni_vslam node
add_executable(${PROJECT_NAME}_node main/omni_vslam_node.cpp  )

target_link_libraries(${PROJECT_NAME}_node ${PROJECT_NAME} ${catkin_LIBRARIES} ${OpenCV_LIBRARYS})
#message(STATUS "-----------------------------------------" ${OpenCV_LIBRARYS})
target_link_libraries(${PROJECT_NAME}_node ${PROJECT_NAME} ${BOOST_SERIALIZATION}
 "/usr/lib/x86_64-linux-gnu/libstdc++.so.6" "/usr/lib/x86_64-linux-gnu/libtiff.so.5"
)

#############
## Install ##
#############

# all install targets should use catkin DESTINATION variables
# See http://ros.org/doc/api/catkin/html/adv_user_guide/variables.html

## Mark executable scripts (Python etc.) for installation
## in contrast to setup.py, you can choose the destination
# install(PROGRAMS
#   scripts/my_python_script
#   DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
# )

## Mark executables and/or libraries for installation
 install(TARGETS ${PROJECT_NAME}_node ${PROJECT_NAME}  
   # g2o models DBoW3
   ARCHIVE DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
   LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
   RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
 ) 

## Mark cpp header files for installation
install(DIRECTORY config/ 
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
  FILES_MATCHING PATTERN "*.*"
  PATTERN ".svn" EXCLUDE
 )

 ## Mark cpp header files for installation
install(DIRECTORY Vocabulary/
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
  FILES_MATCHING PATTERN "*.*"
  PATTERN ".svn" EXCLUDE
 )

## Mark other files for installation (e.g. launch and bag files, etc.)
 install(FILES
   launch/omni_vslam.launch  
   DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)

install(FILES
   lib/libDBoW3.so lib/libg2o.so lib/libmodels.so  
   lib/libomni_vslam.so
   #DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
   DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

#############
## Testing ##
#############

## Add gtest based cpp test target and link libraries
# catkin_add_gtest(${PROJECT_NAME}-test test/test_capturefrontviewimage.cpp)
# if(TARGET ${PROJECT_NAME}-test)
#   target_link_libraries(${PROJECT_NAME}-test ${PROJECT_NAME})
# endif()

## Add folders to be run by python nosetests
# catkin_add_nosetests(test)
