cmake_minimum_required(VERSION 3.12)
project(ObjectiveFeasibilityPump)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_BUILD_TYPE Release)

include(FetchContent)

FetchContent_Declare(
    highs
    GIT_REPOSITORY "https://github.com/ERGO-Code/HiGHS"
    GIT_TAG "v1.11.0"
)
FetchContent_MakeAvailable(highs)

FetchContent_Declare(
    Eigen3
    GIT_REPOSITORY "https://gitlab.com/libeigen/eigen.git"
    GIT_TAG "3.4.1"
)
FetchContent_MakeAvailable(Eigen3)

add_library(ObjectiveFeasibilityPump STATIC src/ObjectiveFeasibilityPump.cpp)
target_link_libraries(ObjectiveFeasibilityPump PRIVATE highs Eigen3::Eigen)
target_include_directories(ObjectiveFeasibilityPump PUBLIC
    $<INSTALL_INTERFACE:include>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
)

set(HIGHS_INCLUDE_PATHs
    ${highs_SOURCE_DIR}/highs
    ${highs_BINARY_DIR}
    CACHE INTERNAL "Path to fetched HiGHS headers" FORCE
)

add_subdirectory(test)