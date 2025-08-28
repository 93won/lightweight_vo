cmake_minimum_required(VERSION 3.10)
project(lightweight_vio)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find packages
find_package(OpenCV REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(glfw3 REQUIRED)
find_package(OpenGL REQUIRED)
find_package(GLEW REQUIRED)
find_package(Threads REQUIRED)

# Build Sophus from thirdparty first with tests disabled
set(BUILD_TESTS OFF CACHE BOOL "Build Sophus tests" FORCE)
add_subdirectory(thirdparty/Sophus)

# Build Ceres from thirdparty without finding system version
set(CERES_DIR ${CMAKE_CURRENT_SOURCE_DIR}/thirdparty/ceres-solver)
set(BUILD_EXAMPLES OFF CACHE BOOL "Build Ceres examples" FORCE)
set(BUILD_BENCHMARKS OFF CACHE BOOL "Build Ceres benchmarks" FORCE)
set(BUILD_TESTING OFF CACHE BOOL "Build Ceres tests" FORCE)
set(SUITESPARSE OFF CACHE BOOL "Enable SuiteSparse" FORCE)
add_subdirectory(${CERES_DIR} EXCLUDE_FROM_ALL)

# Find GLU for legacy OpenGL functions
find_library(GLU_LIBRARY GLU REQUIRED)

# Include directories
include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(${EIGEN3_INCLUDE_DIR})
include_directories(src)
include_directories(thirdparty/imgui)
include_directories(thirdparty/imgui/backends)
include_directories(thirdparty/Sophus)
include_directories(thirdparty/ceres-solver/include)
include_directories(${CMAKE_BINARY_DIR}/thirdparty/ceres-solver/include)

# Collect source files
file(GLOB_RECURSE SOURCES 
    "src/database/*.cpp"
    "src/module/*.cpp"
    "src/util/*.cpp"
    "src/viewer/*.cpp"
)

# ImGui sources
set(IMGUI_SOURCES
    thirdparty/imgui/imgui.cpp
    thirdparty/imgui/imgui_demo.cpp
    thirdparty/imgui/imgui_draw.cpp
    thirdparty/imgui/imgui_tables.cpp
    thirdparty/imgui/imgui_widgets.cpp
    thirdparty/imgui/backends/imgui_impl_glfw.cpp
    thirdparty/imgui/backends/imgui_impl_opengl2.cpp
)



# Add VIO with ImGui viewer
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
    src/viewer/ImGuiViewer.cpp
    ${IMGUI_SOURCES}
)

# Link libraries
# target_link_libraries(test_imgui_viewer glfw OpenGL::GL ${GLU_LIBRARY} GLEW::GLEW Threads::Threads)
target_link_libraries(test_vio_viewer 
    ${OpenCV_LIBS} 
    yaml-cpp 
    glfw 
    OpenGL::GL 
    ${GLU_LIBRARY} 
    GLEW::GLEW 
    Threads::Threads
    ceres
    Sophus::Sophus
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
