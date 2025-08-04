# ===========================================================================
#  Makefile for rwcap (Singularity container version)
#  ---------------------------------------------------------------
#  • Builds the confidential core as a shared library (libcore_rwcap.so)
#  • Builds the front‑end executable (rwcap.exe) that links against it
#  • Keeps all implementation inside the library – only headers are public
# ---------------------------------------------------------------------------

# ----- Toolchain -----------------------------------------------------------
CXX      := g++
CXX_RPATH := 

# ----- Container paths (matching CMakeLists.txt) --------------------------
CUDA_HOME      := /usr/local/cuda
TORCH_DIR      := /usr/local/lib/python3.12/dist-packages/torch
TENSORRT_DIR   := /usr/lib/x86_64-linux-gnu
TORCH_TENSORRT_DIR := /usr/local/lib/python3.12/dist-packages/torch_tensorrt

# ----- External dependency: LibTorch ---------------------------------------
LIBTORCH       := $(TORCH_DIR)
LIBTORCH_INC   := $(LIBTORCH)/include
LIBTORCH_LIB   := $(LIBTORCH)/lib

# ----- DNN Solver paths ----------------------------------------------------
# Updated to use libdnnsolver.a from the specified directory
DNNSOLVER_DIR  := /workspace/TCAD/inference_cpp/build


# ----- Project layout ------------------------------------------------------
PROJECT_ROOT   := $(abspath $(dir $(lastword $(MAKEFILE_LIST))))
# confidential source files
CORE_DIR       := src
# exported headers
PUBLIC_INC_DIR := src/include
# third‑party single‑file deps
DEPS_DIR       := deps
BUILD_DIR      := $(PROJECT_ROOT)/build
OBJ_DIR        := $(BUILD_DIR)/obj
BIN_DIR        := $(PROJECT_ROOT)/bin
LIB_DIR        := $(BUILD_DIR)

# ----- Flags ---------------------------------------------------------------
COMMON_INC  := -I$(PUBLIC_INC_DIR) -I$(DEPS_DIR) \
              -isystem $(DEPS_DIR)/jsoncpp -isystem $(DEPS_DIR)/md5 \
              -I$(LIBTORCH_INC) -I$(LIBTORCH_INC)/torch/csrc/api/include \
              -I$(TORCH_TENSORRT_DIR)/include \
              -I$(CUDA_HOME)/include

COMMON_WARN := -Wall -Wno-conversion -Wno-sign-compare -Wno-unused-function -Wno-unused-parameter

# Match the CMakeLists.txt compile definitions
CXXFLAGS_COMMON := -std=c++17 -O3 -DNDEBUG -DEIGEN_DONT_PARALLELIZE \
                   -fopenmp -D_GLIBCXX_USE_CXX11_ABI=1 \
                   -DHAVE_TORCH_TENSORRT -DTORCH_TENSORRT_ENABLED \
                   $(COMMON_WARN) $(COMMON_INC)

CXXFLAGS_PIC    := $(CXXFLAGS_COMMON) -fPIC -MMD -MP

# Library paths and libs (matching CMakeLists.txt order)
LDLIBS_TORCH    := -L$(LIBTORCH_LIB) \
                   -ltorch -ltorch_cpu -ltorch_cuda -lc10 -lc10_cuda

LDLIBS_TORCH_TENSORRT := -L$(TORCH_TENSORRT_DIR)/lib -ltorchtrt

LDLIBS_TENSORRT := -L$(TENSORRT_DIR) \
                   -lnvinfer -lnvonnxparser -lnvinfer_plugin

LDLIBS_CUDA     := -L$(CUDA_HOME)/lib64 \
                   -lcudart -lcublas -lcurand -lcusolver -lcusparse

LDLIBS_SYSTEM   := -ldl -lpthread -lprotobuf

# RPATH settings (matching CMakeLists.txt) - removed DNNSOLVER_DIR since static lib doesn't need it
LDFLAGS_RPATH   := -Wl,-rpath,$(TORCH_TENSORRT_DIR)/lib \
                   -Wl,-rpath,$(LIBTORCH_LIB) \
                   -Wl,-rpath,$(TENSORRT_DIR) \
                   -Wl,-rpath,$(CUDA_HOME)/lib64 \
                   -Wl,-rpath,$(LIB_DIR) \
                   -Wl,-rpath,$(PROJECT_ROOT) \
                   -Wl,-rpath,'$$ORIGIN/..' \
                   -Wl,-rpath,'$$ORIGIN/../lib' \
                   -Wl,--no-as-needed \
                   -Wl,--allow-shlib-undefined

# ----- Sources & Objects ---------------------------------------------------
CORE_SRCS := $(filter-out $(CORE_DIR)/main.cpp, $(wildcard $(CORE_DIR)/*.cpp))
CORE_OBJS := $(patsubst $(CORE_DIR)/%.cpp,$(OBJ_DIR)/%.o,$(CORE_SRCS))

# single‑file third‑party deps we compile into the library for convenience
DEPS_OBJS := $(OBJ_DIR)/jsoncpp.o $(OBJ_DIR)/md5.o

ALL_OBJS  := $(CORE_OBJS) $(DEPS_OBJS)

# ----- Targets -------------------------------------------------------------
LIB_NAME  := librwcapall.so
LIB_FILE  := $(LIB_DIR)/$(LIB_NAME)
APP_NAME  := rwcap
APP_FILE  := $(BIN_DIR)/$(APP_NAME)

# ---------------------------------------------------------------------------
.PHONY: all clean dirs check-deps print-config

all: check-deps $(APP_FILE)

# Print configuration for debugging
print-config:
	@echo "=== Build Configuration ==="
	@echo "TORCH_DIR: $(TORCH_DIR)"
	@echo "TENSORRT_DIR: $(TENSORRT_DIR)"
	@echo "TORCH_TENSORRT_DIR: $(TORCH_TENSORRT_DIR)"
	@echo "DNNSOLVER_DIR: $(DNNSOLVER_DIR)"
	@echo "=========================="

# Check that dependencies exist - updated to check for .a file
check-deps: print-config
	@if [ ! -f "$(DNNSOLVER_DIR)/libdnnsolver.a" ]; then \
		echo "Error: libdnnsolver.a not found at $(DNNSOLVER_DIR)"; \
		echo "Please update DNNSOLVER_DIR in Makefile"; \
		exit 1; \
	fi
	@if [ ! -d "$(LIBTORCH_LIB)" ]; then \
		echo "Error: LibTorch not found at $(LIBTORCH_LIB)"; \
		exit 1; \
	fi
	@echo "Dependencies check passed ✓"

# Create build directories
dirs:
	mkdir -p $(OBJ_DIR) $(BIN_DIR) $(LIB_DIR)

# ---- Compile rules --------------------------------------------------------
# Core sources -> PIC objects
$(OBJ_DIR)/%.o: $(CORE_DIR)/%.cpp | dirs
	$(CXX) $(CXXFLAGS_PIC) -c $< -o $@

# Third‑party single translation units
$(OBJ_DIR)/jsoncpp.o: $(DEPS_DIR)/jsoncpp/jsoncpp.cpp | dirs
	$(CXX) $(CXXFLAGS_PIC) -c $< -o $@

$(OBJ_DIR)/md5.o: $(DEPS_DIR)/md5/md5.cpp | dirs
	$(CXX) $(CXXFLAGS_PIC) -c $< -o $@

# ---- Library --------------------------------------------------------------
$(LIB_FILE): $(ALL_OBJS)
	$(CXX) -shared -o $@ $^ \
		$(LDLIBS_SYSTEM) \
		$(LDFLAGS_RPATH)
	@echo "[lib]  $@ generated"

-include $(ALL_OBJS:$(OBJ_DIR)/%.o=$(OBJ_DIR)/%.d)

# ---- Front‑end executable -------------------------------------------------
SRC_MAIN := src/main.cpp

$(APP_FILE): $(SRC_MAIN) $(LIB_FILE) | dirs
	$(CXX) -D_NOT_CHECK_LICENSE_ $(CXXFLAGS_COMMON) $< \
	      -L$(LIB_DIR) -L. \
	      -lrwcapall $(DNNSOLVER_DIR)/libdnnsolver.a -lrwcap \
	      $(LDLIBS_SYSTEM) \
	      $(LDLIBS_TORCH) \
	      $(LDLIBS_TORCH_TENSORRT) \
	      $(LDLIBS_TENSORRT) \
	      $(LDLIBS_CUDA) \
	      $(LDFLAGS_RPATH) -o $@
	@echo "[bin]  $@ generated"
	@echo "Checking executable dependencies:"
	@ldd $@ | grep -E "(torch|tensorrt|dnnsolver)" || true

# ---- House‑keeping --------------------------------------------------------
clean:
	rm -rf $(BUILD_DIR) $(BIN_DIR)