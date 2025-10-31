cmake_minimum_required(VERSION 3.15...3.27)

option(BUILD_PYTHON_BINDINGS "Build Python bindings for ObjectiveFeasibilityPump" OFF)
message(STATUS "Building Python Bindings: ${BUILD_PYTHON_BINDINGS}")

project(
    objective_feasibility_pump
    VERSION 0.0.1
    LANGUAGES CXX
)

if (BUILD_PYTHON_BINDINGS)
    find_package(Python REQUIRED COMPONENTS Interpreter Development.Module)
    find_package(pybind11 CONFIG REQUIRED)
    set(TARGET_NAME _core)
else()
    set(TARGET_NAME objective_feasibility_pump)
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
    python_add_library(${TARGET_NAME} MODULE python/src/objective_feasibility_pump_py.cpp
        src/ObjectiveFeasibilityPump.cpp
        WITH_SOABI
    )
    target_link_libraries(${TARGET_NAME} PRIVATE highs Eigen3::Eigen pybind11::headers)
    target_compile_definitions(${TARGET_NAME} PRIVATE VERSION_INFO=${PROJECT_VERSION})

    set_target_properties(${TARGET_NAME} PROPERTIES
            INSTALL_RPATH "$ORIGIN"
            BUILD_WITH_INSTALL_RPATH TRUE
    )

    install(TARGETS highs DESTINATION objective_feasibility_pump)
    install(TARGETS ${TARGET_NAME} DESTINATION objective_feasibility_pump)
    
else() # C++ build
    add_library(${TARGET_NAME} STATIC src/ObjectiveFeasibilityPump.cpp)
    target_link_libraries(${TARGET_NAME} PRIVATE highs)
    target_link_libraries(${TARGET_NAME} PUBLIC Eigen3::Eigen)

    set(HIGHS_INCLUDE_PATHs
        ${highs_SOURCE_DIR}/highs
        ${highs_BINARY_DIR}
        CACHE INTERNAL "Path to fetched HiGHS headers" FORCE
    )

    add_subdirectory(test)
endif()

target_include_directories(${TARGET_NAME} PUBLIC
        $<INSTALL_INTERFACE:include>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
)

target_compile_features(${TARGET_NAME}
        PUBLIC cxx_std_17
)

