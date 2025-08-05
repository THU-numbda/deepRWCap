#include "dnn_solver.h"
#include <iostream>
#include <chrono>
#include <vector>
#include <random>

int main() {
    std::cout << "=== DNN Solver Demo ===" << std::endl;
    std::cout << "Torch version: " << TORCH_VERSION << std::endl;

    // Define Green's function sampler
    DNNSolverGreens greensSolver;
    greensSolver.loadModels(true);

    DNNSolverGrad gradientSolver;
    gradientSolver.loadModels(true);
        
    return 0;
}