cmake_minimum_required(VERSION 3.12)
project(test_feas_pump)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_BUILD_TYPE Release)

include_directories(
    ../include
)

include(FetchContent)

FetchContent_Declare(
    highs
    GIT_REPOSITORY "https://github.com/ERGO-Code/HiGHS"
    GIT_TAG "v1.11.0"
)
FetchContent_MakeAvailable(highs)

FetchContent_Declare(
    eigen
    GIT_REPOSITORY "https://gitlab.com/libeigen/eigen.git"
    GIT_TAG "5.0.0"
)
FetchContent_MakeAvailable(eigen)

add_executable(test_feas_pump
    test_feas_pump.cpp
    ../src/ObjectiveFeasibilityPump.cpp
)
target_link_libraries(test_feas_pump PRIVATE highs eigen)