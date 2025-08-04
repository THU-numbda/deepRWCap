#include "dnn_solver.h"
#include <torch/torch.h>
#include <vector>
#include <thread>
#include "cuda_kernels.h"

DNNSolverGrad::DNNSolverGrad(int device_idx) : DNNSolver(device_idx) {
    // Constructor - base class handles device setup
}

void DNNSolverGrad::loadModels(bool verbose) {
    torch::InferenceMode guard;  // Enable inference mode for this method

    // Stream is already set by calling function, but ensure it's set
    setCurrentStream();


    // Load model paths
    auto modelPaths = loadModelPaths(MODELS_PATH);
    std::string gradientFaceSelectorWeightPredictorPath = modelPaths[2];
    std::string gradientFace2PredictorWithSignPath = modelPaths[3];
    std::string gradientFace1PredictorPath = modelPaths[4];

    // Load models
    gradientFaceSelectorWeightPredictor = loadModel(gradientFaceSelectorWeightPredictorPath, verbose);
    gradientFace2PredictorWithSign = loadModel(gradientFace2PredictorWithSignPath, verbose);
    gradientFace1Predictor = loadModel(gradientFace1PredictorPath, verbose);

    // Move models to device
    gradientFaceSelectorWeightPredictor.to(device);
    gradientFace2PredictorWithSign.to(device);
    gradientFace1Predictor.to(device);

    // Set to evaluation mode
    gradientFaceSelectorWeightPredictor.eval();
    gradientFace2PredictorWithSign.eval();
    gradientFace1Predictor.eval();
}

std::vector<float> DNNSolverGrad::sample(const std::vector<float>& layerStructures, 
                                        const std::vector<float>& cuboidStructures) {
    torch::InferenceMode guard;  // Enable inference mode for this method
    // Stream is already set by calling function, but ensure it's set
    setCurrentStream();
    // For gradient solver, axis is required - this should throw an error or provide default
    throw std::runtime_error("Gradient solver requires axis parameter");
}

std::vector<float> DNNSolverGrad::sample(const std::vector<float>& layerStructures, 
                                        const std::vector<float>& cuboidStructures, 
                                        const std::vector<int>& axis) {
    torch::InferenceMode guard;  // Enable inference mode for this method
    // Stream is already set by calling function, but ensure it's set
    setCurrentStream();
    int batch_size = layerStructures.size() / (MAX_LAYERS * 2);

    auto [dielectricTensor, max_vals] = structureConstruction(layerStructures, cuboidStructures, batch_size, true);
    dielectricTensor = dielectricTensor.view({batch_size, 1, 1, N, N, N});
    
    auto axisTensor = torch::from_blob(const_cast<int*>(axis.data()), {batch_size},
        torch::TensorOptions().dtype(torch::kInt32)).to(device).to(torch::kUInt8);

    auto positionsAndWeights = sampleGradient(dielectricTensor, max_vals, axisTensor);

    auto cpuTensor = positionsAndWeights.to(torch::kCPU).contiguous();
    
    std::vector<float> vec(
        cpuTensor.data_ptr<float>(),
        cpuTensor.data_ptr<float>() + cpuTensor.numel()
    );

    return vec;
}

torch::Tensor DNNSolverGrad::predictGradient(torch::Tensor& dielectricTensor, torch::Tensor& faceIds, torch::Tensor& axis) {
    torch::InferenceMode guard;  // Enable inference mode for this method
    // Stream is already set by calling function, but ensure it's set
    setCurrentStream();

    int batch_size = dielectricTensor.size(0);
    
    // Use pre-allocated gradient buffer instead of torch::zeros
    auto gradient = gradient_buffer_1_.slice(0, 0, batch_size).view({batch_size, N, N});
    auto rotated_tensor = rotate_faces_gradient_launcher(dielectric_buffer_2_, dielectricTensor, faceIds).view({batch_size, 1, 1, N, N, N});
    
    auto face01_mask = (faceIds == 0) | (faceIds == 1);
    auto face01_indices = torch::nonzero(face01_mask).view(-1);
    auto face_other_indices = torch::nonzero(~face01_mask).view(-1);
    
    if (face01_indices.numel() > 0) {
        auto subset1 = rotated_tensor.index_select(0, face01_indices);
        auto g1 = gradientFace1Predictor.forward({subset1}).toTensor();
        gradient.index_copy_(0, face01_indices, g1);
    }
    if (face_other_indices.numel() > 0) {
        auto subset2 = rotated_tensor.index_select(0, face_other_indices);
        auto g2 = gradientFace2PredictorWithSign.forward({subset2}).toTensor();
        gradient.index_copy_(0, face_other_indices, g2);
    }
    
    // Flatten the gradient for post-processing
    auto gradient_flat = gradient.view({batch_size, NN});
    
    // Use gradient_buffer_2_ for post-processing output
    auto processed_gradient = post_process_gradient_launcher(gradient_buffer_2_, gradient_flat, faceIds, axis);
    
    return processed_gradient;
}


torch::Tensor DNNSolverGrad::sampleGradient(torch::Tensor& dielectricTensor, torch::Tensor& max_vals, torch::Tensor& axis) {
    torch::InferenceMode guard;  // Enable inference mode for this method
    // Stream is already set by calling function, but ensure it's set
    setCurrentStream();
    
    // Rotate dielectric tensor based on the axis
    for (int ax = 0; ax < 2; ++ax) {
        auto mask = (axis == ax);
        auto nonz = mask.nonzero();
        if (nonz.numel() > 0) {
            auto idx = nonz.squeeze(1).to(torch::kLong);
            auto subset = dielectricTensor.index_select(0, idx);
            if (ax == 0)      subset = subset.rot90(-1, /*dims=*/{-3, -1});
            else if (ax == 1) subset = subset.rot90(-1, /*dims=*/{-3, -2});
            dielectricTensor.index_copy_(0, idx, subset);
        }
    }
    
    auto fpw = gradientFaceSelectorWeightPredictor.forward({dielectricTensor}).toTensor();
    
    auto faceProbabilities = fpw.slice(1, 0, 6);
    auto selectedFaceTensor = torch::multinomial(faceProbabilities, 1, false);
    auto selectedFacesUint8 = selectedFaceTensor.view({-1}).to(torch::kUInt8);

    auto gradient = predictGradient(dielectricTensor, selectedFacesUint8, axis);
    auto absGradient = torch::abs(gradient);
    auto sampleGradientTensor = torch::multinomial(absGradient, 1, false).view(-1);
    auto sampleGradientInt16 = sampleGradientTensor.to(torch::kInt16);
    
    auto positions = locate_index_cuda_launcher(sampleGradientInt16, selectedFacesUint8, axis);


    auto weights = fpw.slice(1, 6, 7);

    // Get sign of sampled gradient using gather (more efficient than batch indexing)
    auto sign = torch::sign(gradient);
    auto sampleSign = torch::gather(sign, 1, sampleGradientTensor.unsqueeze(1));

    auto centerDielectric = dielectricTensor
        .select(1, 0)      // select channel 0
        .select(1, 0)      // select first element of next dimension
        .select(1, N/2)    // select center x
        .select(1, N/2)    // select center y
        .select(1, N/2)    // select center z
        .view({-1, 1}) * max_vals.view({-1, 1});

    // Single concatenation operation
    auto positionsAndWeightsAndCenter = torch::cat({
        positions, 
        weights * sampleSign, 
        centerDielectric
    }, 1);
        
    return positionsAndWeightsAndCenter;
}