cmake_minimum_required(VERSION 3.10)
project(lightweight_vio)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Set OpenGL preference to avoid warnings
set(OpenGL_GL_PREFERENCE GLVND)

# Find packages
find_package(OpenCV REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(OpenGL REQUIRED)
find_package(GLEW REQUIRED)
find_package(Threads REQUIRED)

# Build Pangolin from thirdparty
set(BUILD_PANGOLIN_EXAMPLES OFF CACHE BOOL "Build Pangolin examples" FORCE)
set(BUILD_PANGOLIN_TESTS OFF CACHE BOOL "Build Pangolin tests" FORCE)
set(BUILD_PANGOLIN_TOOLS OFF CACHE BOOL "Build Pangolin tools" FORCE)
add_subdirectory(thirdparty/pangolin EXCLUDE_FROM_ALL)

# Build Sophus from thirdparty first with tests disabled
set(BUILD_TESTS OFF CACHE BOOL "Build Sophus tests" FORCE)
set(BUILD_SOPHUS_TESTS OFF CACHE BOOL "Build Sophus tests" FORCE)
set(BUILD_SOPHUS_EXAMPLES OFF CACHE BOOL "Build Sophus examples" FORCE)
add_subdirectory(thirdparty/Sophus)

# Build spdlog from thirdparty
set(SPDLOG_BUILD_EXAMPLES OFF CACHE BOOL "Build spdlog examples" FORCE)
set(SPDLOG_BUILD_TESTS OFF CACHE BOOL "Build spdlog tests" FORCE)
set(SPDLOG_BUILD_BENCH OFF CACHE BOOL "Build spdlog benchmarks" FORCE)
add_subdirectory(thirdparty/spdlog)

# Build Ceres from thirdparty without finding system version
set(CERES_DIR ${CMAKE_CURRENT_SOURCE_DIR}/thirdparty/ceres-solver)
set(BUILD_EXAMPLES OFF CACHE BOOL "Build Ceres examples" FORCE)
set(BUILD_BENCHMARKS OFF CACHE BOOL "Build Ceres benchmarks" FORCE)
set(BUILD_TESTING OFF CACHE BOOL "Build Ceres tests" FORCE)
set(SUITESPARSE OFF CACHE BOOL "Enable SuiteSparse" FORCE)
set(PROVIDE_UNINSTALL_TARGET OFF CACHE BOOL "Disable Ceres uninstall target" FORCE)

# Prevent finding system Ceres
set(CMAKE_DISABLE_FIND_PACKAGE_Ceres ON)

add_subdirectory(${CERES_DIR} EXCLUDE_FROM_ALL)

# Find GLU for legacy OpenGL functions
find_library(GLU_LIBRARY GLU REQUIRED)

# Include directories
include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(${EIGEN3_INCLUDE_DIR})
include_directories(src)
include_directories(thirdparty/pangolin/include)
include_directories(${CMAKE_BINARY_DIR}/thirdparty/pangolin/include)
include_directories(thirdparty/imgui/backends)
include_directories(thirdparty/Sophus)
include_directories(thirdparty/spdlog/include)
include_directories(thirdparty/ceres-solver/include)
include_directories(${CMAKE_BINARY_DIR}/thirdparty/ceres-solver/include)

# Collect source files
file(GLOB_RECURSE SOURCES 
    "src/database/*.cpp"
    "src/module/*.cpp"
    "src/util/*.cpp"
    "src/viewer/*.cpp"
)



# Add VIO with Pangolin viewer
add_executable(test_vio_viewer test_vio_viewer.cpp 
    src/database/Frame.cpp
    src/database/Feature.cpp
    src/database/MapPoint.cpp  
    src/module/FeatureTracker.cpp
    src/module/PoseOptimizer.cpp
    src/module/Estimator.cpp
    # src/factor/SE3Manifolds.cpp
    src/factor/PnPFactors.cpp
    src/factor/Parameters.cpp
    src/util/Config.cpp
    src/util/EurocUtils.cpp
    src/viewer/PangolinViewer.cpp
)

# Link libraries
target_link_libraries(test_vio_viewer 
    ${OpenCV_LIBS} 
    yaml-cpp 
    OpenGL::GL 
    GLEW::GLEW
    Threads::Threads
    ceres
    Sophus::Sophus
    spdlog::spdlog
    pango_display
    pango_windowing
    pango_opengl
    pango_core
    pango_image
)


# Link Eigen3 to all targets
if(TARGET Eigen3::Eigen)
    # target_link_libraries(test Eigen3::Eigen)
    # target_link_libraries(test_imgui_viewer Eigen3::Eigen)
    target_link_libraries(test_vio_viewer Eigen3::Eigen)
else()
    # Fallback for older CMake versions
    # target_include_directories(test PRIVATE ${EIGEN3_INCLUDE_DIR})
    # target_include_directories(test_imgui_viewer PRIVATE ${EIGEN3_INCLUDE_DIR})
    target_include_directories(test_vio_viewer PRIVATE ${EIGEN3_INCLUDE_DIR})
endif()

# Print OpenCV information for debugging
message(STATUS "OpenCV library status:")
message(STATUS "    version: ${OpenCV_VERSION}")
message(STATUS "    libraries: ${OpenCV_LIBS}")
message(STATUS "    include path: ${OpenCV_INCLUDE_DIRS}")
