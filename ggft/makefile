CXXFLAGS = -std=c++17 -O3 -DNDEBUG -march=native -fopenmp


CXX = g++

EIGEN_DIR = dep/eigen-3.4.0

DIR = src
OUTDIR = bin

all: ggft

ggft: $(DIR)/main.cpp $(DIR)/ggft_solver.cpp
	mkdir -p $(OUTDIR)
	$(CXX) $(DIR)/main.cpp $(DIR)/ggft_solver.cpp -o $(OUTDIR)/ggft -I$(EIGEN_DIR) -I$(DIR) $(CXXFLAGS)

clean:
	rm $(OUTDIR)/ggft