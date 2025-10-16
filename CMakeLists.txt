cmake_minimum_required(VERSION 3.12)
project(test_highs)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_BUILD_TYPE Debug)

include_directories(
    ./extern/eigen
    include
)
add_subdirectory(./extern/HiGHS)

add_executable(test_highs
    test_highs.cpp
)

add_executable(test_feas_pump
    test_feas_pump.cpp
    src/ObjectiveFeasibilityPump.cpp
)

target_link_libraries(test_highs PRIVATE highs::highs)
target_link_libraries(test_feas_pump PRIVATE highs::highs)