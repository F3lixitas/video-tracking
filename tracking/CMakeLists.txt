cmake_minimum_required(VERSION 3.26)
project(tracking)

set(CMAKE_CXX_STANDARD 17)

add_executable(tracking main.cpp)

target_link_libraries(tracking PUBLIC opencv_core opencv_highgui opencv_videoio opencv_video opencv_imgproc)
target_include_directories(tracking PUBLIC /usr/local/include/opencv4)