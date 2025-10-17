cmake_minimum_required(VERSION 3.0.2)
project(fmcw_receive)


find_package(catkin REQUIRED)
find_package(PCL REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(Ceres REQUIRED)



find_package(catkin REQUIRED COMPONENTS
  message_generation
  roscpp
  std_msgs
)

add_message_files(FILES
  XiLidarPacket.msg
)


generate_messages(DEPENDENCIES
  std_msgs
)


catkin_package(
  CATKIN_DEPENDS
  message_runtime
  DEPENDS system_lib
)


include_directories(
  include ${catkin_INCLUDE_DIRS}
  include
  ${PCL_INCLUDE_DIRS}
  ${CERES_INCLUDE_DIRS}
# ${catkin_INCLUDE_DIRS}
)

link_directories(
  include
  ${PCL_LIBRARY_DIRS}
  ${CERES_LIBRARY_DIRS}
)


add_executable(decoder
  src/decoder.cpp
)
add_dependencies(decoder ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})
target_link_libraries(decoder
  ${catkin_LIBRARIES}
)

add_executable(driver
  src/driver.cpp
)
add_dependencies(driver ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})
target_link_libraries(driver
  ${catkin_LIBRARIES} 
)


