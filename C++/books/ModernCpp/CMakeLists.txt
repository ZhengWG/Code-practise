cmake_minimum_required(VERSION 3.10)

# 项目名称
project(ModernCpp)

# 设置C++标准
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include_directories("${PROJECT_SOURCE_DIR}/section5")
add_subdirectory(section5)
set(EXTRA_LIBS ${EXTRA_LIBS} section5)