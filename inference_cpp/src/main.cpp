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
    
    // Structure file
    int numSamples = 2;
    std::string structureDataPath = "/workspace/datasets/example.structure";
    std::vector<float> layerStructures;
    std::vector<float> cuboidStructures;
    greensSolver.readStructureFile(structureDataPath, layerStructures, cuboidStructures, numSamples, true);

    // Sample Green's function
    std::vector<float> greensSample = greensSolver.sample(layerStructures, cuboidStructures);

    std::vector<float> gradSample = gradientSolver.sample(layerStructures, cuboidStructures, {0, 1, 2}); // Sample with axis

    std::cout << "Sampled Green's function: " << greensSample << std::endl;
    
    return 0;
}