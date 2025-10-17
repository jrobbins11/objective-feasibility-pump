cmake_minimum_required(VERSION 3.15...3.27)

option(BUILD_PYTHON_BINDINGS "Build Python bindings for ObjectiveFeasibilityPump" OFF)
message(STATUS "Building Python Bindings: ${BUILD_PYTHON_BINDINGS}")

project(
    objective_feasibility_pump
    VERSION 0.0.1
    LANGUAGES CXX
)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_BUILD_TYPE Release)

if (BUILD_PYTHON_BINDINGS)
    find_package(Python REQUIRED COMPONENTS Interpreter Development.Module)
    find_package(pybind11 CONFIG REQUIRED)
endif()


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

if (BUILD_PYTHON_BINDINGS)
    python_add_library(_core MODULE python/src/objective_feasibility_pump_py.cpp
        src/ObjectiveFeasibilityPump.cpp
        WITH_SOABI
    )
    target_link_libraries(_core PRIVATE highs Eigen3::Eigen pybind11::headers)
    target_compile_definitions(_core PRIVATE VERSION_INFO=${PROJECT_VERSION})
    target_include_directories(_core PUBLIC
            $<INSTALL_INTERFACE:include>
            $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    )
    install(TARGETS _core DESTINATION objective_feasibility_pump)
    
else() # C++ build
    add_library(objective_feasibility_pump STATIC src/ObjectiveFeasibilityPump.cpp)
    target_link_libraries(objective_feasibility_pump PRIVATE highs Eigen3::Eigen)
    target_include_directories(objective_feasibility_pump PUBLIC
        $<INSTALL_INTERFACE:include>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    )

    set(HIGHS_INCLUDE_PATHs
        ${highs_SOURCE_DIR}/highs
        ${highs_BINARY_DIR}
        CACHE INTERNAL "Path to fetched HiGHS headers" FORCE
    )

    add_subdirectory(test)
endif()

