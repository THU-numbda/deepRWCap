#include "dnn_solver.h"
#include <torch/torch.h>
#include <vector>
#include <thread>
#include "cuda_kernels.h"

DNNSolverGreens::DNNSolverGreens(int device_idx) : DNNSolver(device_idx) {
    // Constructor - base class handles device setup
}

void DNNSolverGreens::loadModels(bool verbose) {
    torch::InferenceMode guard;  // Enable inference mode for this method
    
    // Set current stream for all operations
    setCurrentStream();

    // Load model paths
    auto modelPaths = loadModelPaths(MODELS_PATH);
    std::string faceSelectorPath = modelPaths[0];
    std::string greenPredictorPath = modelPaths[1];

    // Load models
    faceSelector = loadModel(faceSelectorPath, verbose);
    greenPredictor = loadModel(greenPredictorPath, verbose);

    // Move models to device
    faceSelector.to(device);
    greenPredictor.to(device);

    // Set to evaluation mode
    faceSelector.eval();
    greenPredictor.eval();
}

std::vector<float> DNNSolverGreens::sample(const std::vector<float>& layerStructures,
                                           const std::vector<float>& cuboidStructures) {
    torch::InferenceMode guard;
    setCurrentStream();
    
    int batch_size = layerStructures.size() / (MAX_LAYERS * 2);
    auto [dielectricTensor, max_vals] = structureConstruction(layerStructures, cuboidStructures, batch_size, true);
    dielectricTensor = dielectricTensor.view({batch_size, 1, 1, N, N, N});
    
    auto positions = sampleGreensFunction(dielectricTensor);
    
    // For better performance, consider pinned memory for the output
    auto cpu_positions = positions.to(torch::kCPU, /*non_blocking=*/true);
    
    // Ensure transfer is complete before accessing data
    if (device.is_cuda()) {
        c10::cuda::getCurrentCUDAStream().synchronize();
    }
    
    // Ensure contiguous memory
    cpu_positions = cpu_positions.contiguous();
    
    std::vector<float> vec(
        cpu_positions.data_ptr<float>(),
        cpu_positions.data_ptr<float>() + cpu_positions.numel()
    );
    
    return vec;
}

std::vector<float> DNNSolverGreens::sample(const std::vector<float>& layerStructures, 
                                          const std::vector<float>& cuboidStructures, 
                                          const std::vector<int>& axis) {
    torch::InferenceMode guard;  // Enable inference mode for this method
    
    // Set current stream for all operations
    setCurrentStream();
    
    // Issue warning only once per program execution
    static bool warning_issued = false;
    if (!warning_issued) {
        std::cerr << "Warning: Axis parameter provided to Green's function solver but will be ignored. "
                  << "Green's function sampling does not use axis information. "
                  << "(This warning will only be shown once)" << std::endl;
        warning_issued = true;
    }
    
    // For Green's function, axis is ignored - just call the version without axis
    return sample(layerStructures, cuboidStructures);
}

torch::Tensor DNNSolverGreens::sampleGreensFunction(torch::Tensor& dielectricTensor) {

    int batch_size = dielectricTensor.size(0);
    
    // Select faces
    auto faceProbabilities = faceSelector.forward({dielectricTensor}).toTensor();
    auto selectedFaceTensor = torch::multinomial(faceProbabilities, 1, false);
    
    // Avoid intermediate view and type conversion - do it in one step
    auto selectedFacesUint8 = selectedFaceTensor.to(torch::kUInt8).view({-1});
    
    // Greens function sampling - avoid redundant view at the end
    auto rotated_tensor = rotate_faces_greens_launcher(dielectric_buffer_2_, 
                                                       dielectricTensor, 
                                                       selectedFacesUint8);
    
    // Pass the already correctly shaped tensor
    auto greensFunction = greenPredictor.forward({rotated_tensor.view({batch_size, 1, 1, N, N, N})})
                                       .toTensor()
                                       .view({batch_size, NN});
    
    auto sampleGreensFunctionTensor = torch::multinomial(greensFunction, 1, false)
                                            .to(torch::kInt16)
                                            .view({-1});
    
    return locate_index_simple_cuda_launcher(sampleGreensFunctionTensor, selectedFacesUint8);
}