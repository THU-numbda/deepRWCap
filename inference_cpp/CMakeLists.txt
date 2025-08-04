cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(dnnsolver LANGUAGES CXX CUDA)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Helper function
macro(set_ifndef var val)
    if(NOT DEFINED ${var})
        set(${var} ${val})
    endif()
endmacro()

# Set default values
set_ifndef(CUDA_INSTALL_DIR /usr/local/cuda)
set_ifndef(CMAKE_BUILD_TYPE Release)

# Enable verbose output for debugging
set(CMAKE_VERBOSE_MAKEFILE ON)

# ------------------------------
# 1. CUDA Setup
# ------------------------------
find_package(CUDAToolkit REQUIRED)
message(STATUS "Found CUDA ${CUDAToolkit_VERSION}")

# ------------------------------
# 2. PyTorch Setup
# ------------------------------
set(TORCH_DIR "/usr/local/lib/python3.12/dist-packages/torch")
list(APPEND CMAKE_PREFIX_PATH "${TORCH_DIR}")
find_package(Torch REQUIRED)
message(STATUS "Found PyTorch ${Torch_VERSION}")

# ------------------------------
# 3. Find Protobuf (required by PyTorch)
# ------------------------------
find_package(Protobuf REQUIRED)
message(STATUS "Found Protobuf: ${Protobuf_VERSION}")

# ------------------------------
# 4. torch_tensorrt Setup (MUST come before TensorRT)
# ------------------------------
set(TORCH_TENSORRT_INCLUDE_DIR "/usr/local/lib/python3.12/dist-packages/torch_tensorrt/include")
set(TORCH_TENSORRT_LIB_DIR "/usr/local/lib/python3.12/dist-packages/torch_tensorrt/lib")

# Check for torch_tensorrt availability
if(EXISTS "${TORCH_TENSORRT_INCLUDE_DIR}/torch_tensorrt/torch_tensorrt.h")
    set(HAVE_TORCH_TENSORRT TRUE)
    message(STATUS "torch_tensorrt C++ API found")
    
    # ONLY use the main runtime library to avoid duplicate registration
    set(TORCH_TENSORRT_LIBRARIES "${TORCH_TENSORRT_LIB_DIR}/libtorchtrt.so")
    
    if(NOT EXISTS ${TORCH_TENSORRT_LIBRARIES})
        # If libtorchtrt.so doesn't exist, try the runtime library only
        set(TORCH_TENSORRT_LIBRARIES "${TORCH_TENSORRT_LIB_DIR}/libtorchtrt_runtime.so")
        if(NOT EXISTS ${TORCH_TENSORRT_LIBRARIES})
            message(FATAL_ERROR "No suitable torch_tensorrt library found")
        endif()
    endif()
    
    message(STATUS "Using torch_tensorrt library: ${TORCH_TENSORRT_LIBRARIES}")
else()
    message(FATAL_ERROR "torch_tensorrt C++ API not found")
endif()

# ------------------------------
# 5. TensorRT Setup
# ------------------------------
set(TENSORRT_INCLUDE_DIR "/usr/include/x86_64-linux-gnu")
set(TENSORRT_LIB_DIR "/usr/lib/x86_64-linux-gnu")

# Use the EXACT library paths
set(TENSORRT_NVINFER_LIB "${TENSORRT_LIB_DIR}/libnvinfer.so")
set(TENSORRT_NVONNXPARSER_LIB "${TENSORRT_LIB_DIR}/libnvonnxparser.so") 
set(TENSORRT_NVINFER_PLUGIN_LIB "${TENSORRT_LIB_DIR}/libnvinfer_plugin.so")

# Verify these exact files exist
foreach(lib ${TENSORRT_NVINFER_LIB} ${TENSORRT_NVONNXPARSER_LIB} ${TENSORRT_NVINFER_PLUGIN_LIB})
    if(NOT EXISTS ${lib})
        message(FATAL_ERROR "TensorRT library not found: ${lib}")
    endif()
endforeach()

set(TENSORRT_LIBRARIES
    ${TENSORRT_NVINFER_LIB}
    ${TENSORRT_NVONNXPARSER_LIB}
    ${TENSORRT_NVINFER_PLUGIN_LIB}
)

message(STATUS "TensorRT libraries found")

# ------------------------------
# 6. OpenMP
# ------------------------------
find_package(OpenMP REQUIRED)

# ------------------------------
# 7. Source Files
# ------------------------------
set(SRC_DIR "${CMAKE_SOURCE_DIR}/src")

# Library sources (excluding main.cpp)
file(GLOB_RECURSE LIBRARY_SOURCES 
    "${SRC_DIR}/*.cpp" 
    "${SRC_DIR}/*.cu"
)
list(FILTER LIBRARY_SOURCES EXCLUDE REGEX ".*main\\.cpp$")

# Executable source
set(EXECUTABLE_SOURCES "${SRC_DIR}/main.cpp")

# ------------------------------
# 8. Create Shared Library Target
# ------------------------------
add_library(dnnsolver SHARED ${LIBRARY_SOURCES})

# Set properties
set_target_properties(dnnsolver PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
    POSITION_INDEPENDENT_CODE ON
    CUDA_SEPARABLE_COMPILATION ON
    ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)

# Include directories - order matters!
target_include_directories(dnnsolver PUBLIC
    ${SRC_DIR}
    ${TORCH_TENSORRT_INCLUDE_DIR}    # torch_tensorrt first
    ${TORCH_INCLUDE_DIRS}            # PyTorch includes
    ${TENSORRT_INCLUDE_DIR}          # TensorRT includes
    ${CUDA_INSTALL_DIR}/include      # CUDA includes
    ${Protobuf_INCLUDE_DIRS}         # Protobuf includes
)

# Compile options - compatible with PyTorch
target_compile_options(dnnsolver PUBLIC
    ${TORCH_CXX_FLAGS}
    $<$<CONFIG:Debug>:-g>
    $<$<CONFIG:Release>:-O3>
)

# Compile definitions
target_compile_definitions(dnnsolver PUBLIC
    _GLIBCXX_USE_CXX11_ABI=1
    HAVE_TORCH_TENSORRT
    TORCH_TENSORRT_ENABLED
)

# Link libraries - ORDER IS CRITICAL!
# First Protobuf (required by PyTorch)
target_link_libraries(dnnsolver PUBLIC ${Protobuf_LIBRARIES})

# Then PyTorch
target_link_libraries(dnnsolver PUBLIC ${TORCH_LIBRARIES})

# Then torch_tensorrt (extends PyTorch's deserializer)
target_link_libraries(dnnsolver PUBLIC ${TORCH_TENSORRT_LIBRARIES})

# Then TensorRT
target_link_libraries(dnnsolver PUBLIC ${TENSORRT_LIBRARIES})

# Then CUDA and other dependencies
target_link_libraries(dnnsolver PUBLIC
    CUDA::cudart
    CUDA::cublas
    CUDA::curand
    CUDA::cusolver
    CUDA::cusparse
    OpenMP::OpenMP_CXX
    ${CMAKE_DL_LIBS}
    pthread
)

# Link options for proper symbol resolution
if(NOT MSVC)
    target_link_options(dnnsolver PUBLIC
        -Wl,--no-as-needed              # Force linking all libraries
        -Wl,--allow-shlib-undefined     # Allow undefined symbols in shared libs
    )
endif()

# ------------------------------
# 9. Create Executable Target
# ------------------------------
add_executable(dnnsolver_exec ${EXECUTABLE_SOURCES})

# Set properties
set_target_properties(dnnsolver_exec PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
    OUTPUT_NAME dnnsolver
    ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)

# Include directories for executable
target_include_directories(dnnsolver_exec PRIVATE
    ${SRC_DIR}
    ${TORCH_TENSORRT_INCLUDE_DIR}
    ${TORCH_INCLUDE_DIRS}
    ${TENSORRT_INCLUDE_DIR}
    ${CUDA_INSTALL_DIR}/include
    ${Protobuf_INCLUDE_DIRS}
)

# Compile definitions
target_compile_definitions(dnnsolver_exec PRIVATE
    _GLIBCXX_USE_CXX11_ABI=1
    HAVE_TORCH_TENSORRT
    TORCH_TENSORRT_ENABLED
)

# Link executable to library
target_link_libraries(dnnsolver_exec PRIVATE dnnsolver)

# ------------------------------
# 10. RPATH Settings
# ------------------------------
set(RUNTIME_LIBRARY_PATHS
    "${TORCH_TENSORRT_LIB_DIR}"          # torch_tensorrt libs first
    "${TORCH_DIR}/lib"                    # PyTorch libs
    "${TENSORRT_LIB_DIR}"                # TensorRT libs
    "${CUDA_INSTALL_DIR}/lib64"          # CUDA libs
    "$ORIGIN/../lib"                     # Relative path to lib dir
)

# Convert list to colon-separated string
string(REPLACE ";" ":" RUNTIME_LIBRARY_PATH_STRING "${RUNTIME_LIBRARY_PATHS}")

# Set RPATH for both library and executable
set_target_properties(dnnsolver dnnsolver_exec PROPERTIES
    BUILD_RPATH "${RUNTIME_LIBRARY_PATH_STRING}"
    INSTALL_RPATH "${RUNTIME_LIBRARY_PATH_STRING}"
    BUILD_WITH_INSTALL_RPATH FALSE
    INSTALL_RPATH_USE_LINK_PATH TRUE
)

# ------------------------------
# 11. Installation
# ------------------------------
install(TARGETS dnnsolver_exec dnnsolver
    RUNTIME DESTINATION bin
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
)

# Install headers if needed
install(DIRECTORY ${SRC_DIR}/
    DESTINATION include/dnnsolver
    FILES_MATCHING PATTERN "*.h" PATTERN "*.hpp"
)

# ------------------------------
# 12. Print Summary
# ------------------------------
message(STATUS "================================")
message(STATUS "DNNSolver Build Configuration:")
message(STATUS "  - Build Type: ${CMAKE_BUILD_TYPE}")
message(STATUS "  - CUDA: ${CUDAToolkit_VERSION}")
message(STATUS "  - PyTorch: ${Torch_VERSION} at ${TORCH_DIR}")
message(STATUS "  - Protobuf: ${Protobuf_VERSION}")
message(STATUS "  - torch_tensorrt: ENABLED at ${TORCH_TENSORRT_LIB_DIR}")
message(STATUS "  - TensorRT: ${TENSORRT_LIB_DIR}")
message(STATUS "  - OpenMP: FOUND")
message(STATUS "  - Output:")
message(STATUS "    - Executable: ${CMAKE_BINARY_DIR}/bin/dnnsolver")
message(STATUS "    - Library: ${CMAKE_BINARY_DIR}/lib/libdnnsolver.so")
message(STATUS "================================")

# Print linked libraries for debugging
get_target_property(DNNSOLVER_LIBS dnnsolver LINK_LIBRARIES)
message(STATUS "Library dependencies: ${DNNSOLVER_LIBS}")