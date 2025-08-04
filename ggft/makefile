CXXFLAGS = -std=c++17 -O3 -DNDEBUG -march=native -fopenmp

COMPILER_PATH = /data1/huangjc/local/bin/g++
ifeq ($(shell test -e $(COMPILER_PATH) && echo -n yes),yes)
    CXX = $(COMPILER_PATH)
	CXXFLAGS := $(CXXFLAGS) -Wl,-rpath,/data1/huangjc/local/lib64
else
    CXX = g++
endif

# EIGEN_DIR = /data1/huangjc/lib/eigen-3.4.0
EIGEN_DIR = dep/eigen-3.4.0
# ifneq ($(shell test -e $(EIGEN_DIR) && echo -n yes),yes)
#     EIGEN_DIR = /data1/huangjc/tool/eigen-3.4.0
# endif

DIR = src
OUTDIR = bin

all: ggft

ggft: $(DIR)/main.cpp $(DIR)/ggft_solver.cpp
	mkdir -p $(OUTDIR)
	$(CXX) $(DIR)/main.cpp $(DIR)/ggft_solver.cpp -o $(OUTDIR)/ggft -I$(EIGEN_DIR) -I$(DIR) $(CXXFLAGS)

clean:
	rm $(OUTDIR)/ggft

time_dgemm: time_dgemm.c
	gcc -o time_dgemm time_dgemm.c  -I/data1/huangjc/lib/OpenBLAS/include -L/data1/huangjc/lib/OpenBLAS/lib -Wl,-rpath,/data1/huangjc/lib/OpenBLAS/lib -lopenblas -O3

backend_openblas: backend_openblas.cpp
	g++ backend_openblas.cpp -o backend_openblas -O3 -march=native -DEIGEN_USE_BLAS -DEIGEN_USE_LAPACKE -I/data1/huangjc/lib/eigen-3.4.0 -I/data1/huangjc/lib/OpenBLAS/include -L/data1/huangjc/lib/OpenBLAS/lib -lopenblas -Wl,-rpath,/data1/huangjc/lib/OpenBLAS/lib

explict_openblas: explict_openblas.cpp
	g++ explict_openblas.cpp -o explict_openblas -O3 -march=native -I/data1/huangjc/lib/eigen-3.4.0 -I/data1/huangjc/lib/OpenBLAS/include -L/data1/huangjc/lib/OpenBLAS/lib -lopenblas -Wl,-rpath,/data1/huangjc/lib/OpenBLAS/lib

no_openblas: no_openblas.cpp
	g++ no_openblas.cpp -o no_openblas -I/data1/huangjc/lib/eigen-3.4.0  -O3 -march=native

check: check.cpp
	g++ check.cpp -o check -I/data1/huangjc/lib/eigen-3.4.0  -O3 -march=native

check_rwcap: check_rwcap.cpp
	g++ check_rwcap.cpp -o check_rwcap -I/data1/huangjc/lib/eigen-3.4.0  -O3 -march=native
	
check_explict_mkl: check_explict_mkl.cpp
	g++ check_explict_mkl.cpp -o check_explict_mkl -O3 -march=native -I/data1/huangjc/lib/eigen-3.4.0 $(MKL_INC) $(MKL_LIBS)

check_backend_mkl: check_backend_mkl.cpp
	g++ check_backend_mkl.cpp -o check_backend_mkl -O3 -march=native -DEIGEN_USE_BLAS -DEIGEN_USE_LAPACKE -I/data1/huangjc/lib/eigen-3.4.0 $(MKL_INC) $(MKL_LIBS)

explict_mkl: explict_mkl.cpp
	g++ explict_mkl.cpp -o explict_mkl -O3 -march=native -I/data1/huangjc/lib/eigen-3.4.0 $(MKL_INC) $(MKL_LIBS)

backend_mkl: backend_mkl.cpp
	g++ backend_mkl.cpp -o backend_mkl -O3 -march=native -DEIGEN_USE_BLAS -DEIGEN_USE_LAPACKE -I/data1/huangjc/lib/eigen-3.4.0 $(MKL_INC) $(MKL_LIBS)

