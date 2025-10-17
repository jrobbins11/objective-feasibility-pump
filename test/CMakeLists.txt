cmake_minimum_required(VERSION 3.12)
project(test_feas_pump)

add_executable(test_feas_pump
    test_feas_pump.cpp
)
target_link_libraries(test_feas_pump PRIVATE ObjectiveFeasibilityPump)

target_include_directories(test_feas_pump PRIVATE
    ${HIGHS_INCLUDE_PATHs}
)