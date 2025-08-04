#include "dnn_solver.h"
#include <torch/torch.h>
#include <torch/script.h>
#include <torch/nn/utils/rnn.h> // For pad_sequence
#include <vector>
#include <chrono>
#include <fstream>
#include <sstream>
#include <iostream>

#include "cuda_kernels.h"

DNNSolver::DNNSolver(int device_idx) : device(torch::kCPU) {

    // Detect if CUDA is available and set the device accordingly
    if (torch::cuda::is_available()) {
        device = torch::Device(torch::kCUDA, device_idx);
        
        // Create a custom CUDA stream for this solver
        cuda_stream_ = c10::cuda::getStreamFromPool(false, device_idx);

        // Avoid multiple torch threads (not necessary)
        torch::set_num_threads(1);
    } else {
        std::cout << "CUDA is not available. Using CPU.\n";
        device = torch::Device(torch::kCPU);
    }

    // Pre-allocate working buffers
    dielectric_buffer_1_ = torch::empty({MAX_SAMPLES, N, N, N}, 
                                       torch::TensorOptions().dtype(torch::kFloat32).device(device));
    dielectric_buffer_2_ = torch::empty({MAX_SAMPLES, N, N, N}, 
                                       torch::TensorOptions().dtype(torch::kFloat32).device(device));
    gradient_buffer_1_ = torch::empty({MAX_SAMPLES, NN}, 
                                     torch::TensorOptions().dtype(torch::kFloat32).device(device));
    gradient_buffer_2_ = torch::empty({MAX_SAMPLES, NN}, 
                                     torch::TensorOptions().dtype(torch::kFloat32).device(device));
}

void DNNSolver::setCurrentStream() {
    if (device.is_cuda() && cuda_stream_.has_value()) {
        // Set the current CUDA stream for all operations
        c10::cuda::setCurrentCUDAStream(cuda_stream_.value());
    }
}

// Rest of the implementation remains the same as before...
torch::jit::script::Module DNNSolver::loadModel(const std::string& modelPath, bool verbose) {
    torch::jit::script::Module module;
    try {
        // Deserialize the ScriptModule from a file using torch::jit::load().
        module = torch::jit::load(modelPath);
    }
    catch (const c10::Error& e) {
        std::cerr << "Error loading the model from " << modelPath << "\n";
        std::cerr << e.what() << "\n";
        throw;
    }
    if(verbose){
        std::cout << "\tModel loaded successfully from " << modelPath << "\n";
    }
    return module;
}

std::vector<std::string> DNNSolver::loadModelPaths(const std::string& filename) {
    std::vector<std::string> paths;
    std::ifstream file(filename);
    std::string line;
    
    while (std::getline(file, line)) {
        // Skip empty lines and comments
        if (line.empty() || line[0] == '#') continue;
        
        // Extract path before comment
        size_t commentPos = line.find('#');
        std::string path = line.substr(0, commentPos);
        
        // Trim whitespace
        size_t start = path.find_first_not_of(" \t");
        if (start != std::string::npos) {
            size_t end = path.find_last_not_of(" \t");
            paths.push_back(path.substr(start, end - start + 1));
        }
    }
    
    return paths;
}

void DNNSolver::readBinaryFile(const std::string& filePath, 
                              std::vector<float>& dielectricConfigs, 
                              std::vector<float>& targetFunctions, 
                              int numSamples,
                              int structureSize, 
                              DataType inputType,
                              bool verbose) {
    std::ifstream file(filePath, std::ios::binary);
    if (!file) {
        std::cerr << "Error opening file: " << filePath << std::endl;
        throw;
    }
    
    auto start = std::chrono::high_resolution_clock::now();
    
    // Constants
    const int dielectricSize = NNN;
    const int targetSize = NN * 6;
    
    dielectricConfigs.resize(numSamples * dielectricSize);
    targetFunctions.resize(numSamples * targetSize);
    
    // Ignore the first two values (size depends on input type)
    file.ignore(2 * (inputType == DataType::DOUBLE ? sizeof(double) : sizeof(float)));
    
    for (int i = 0; i < numSamples; ++i) {
        if (inputType == DataType::DOUBLE) {
            // Reading from double data
            std::vector<double> dielectricTemp(dielectricSize);
            std::vector<double> targetTemp(targetSize);
            
            // Read dielectric configuration (as doubles)
            file.read(reinterpret_cast<char*>(dielectricTemp.data()), dielectricSize * sizeof(double));
            for (int j = 0; j < dielectricSize; ++j) {
                dielectricConfigs[i * dielectricSize + j] = static_cast<float>(dielectricTemp[j]);
            }
            
            // Ignore structure data (as doubles)
            file.ignore(structureSize * sizeof(double));
            
            // Read target function (as doubles)
            file.read(reinterpret_cast<char*>(targetTemp.data()), targetSize * sizeof(double));
            for (int j = 0; j < targetSize; ++j) {
                targetFunctions[i * targetSize + j] = static_cast<float>(targetTemp[j]);
            }
        } else {
            // Reading from float data directly
            // Read dielectric configuration (as floats)
            file.read(reinterpret_cast<char*>(&dielectricConfigs[i * dielectricSize]), 
                     dielectricSize * sizeof(float));
            
            // Ignore structure data (as floats)
            file.ignore(structureSize * sizeof(float));
            
            // Read target function (as floats)
            file.read(reinterpret_cast<char*>(&targetFunctions[i * targetSize]), 
                     targetSize * sizeof(float));
        }
    }
    
    file.close();
    
    auto end = std::chrono::high_resolution_clock::now();
    double totalTime = std::chrono::duration<double>(end - start).count();
    
    if (verbose) {
        std::cout << "================================================================\n"
                  << "Reading input file" << std::endl;
        std::cout << "\tRead " << dielectricConfigs.size() / dielectricSize 
                  << " dielectric configurations\n\tRead " 
                  << targetFunctions.size() / targetSize << " target functions." << std::endl;
        std::cout << "Time: " << (1000 * totalTime) << " ms" << std::endl;
        std::cout << "================================================================\n" << std::endl;
    }
}

int DNNSolver::readStructureFile(const std::string& filePath, std::vector<float>& layerStructures, std::vector<float>& cuboidStructures, 
                                int numSamples, bool verbose) {
    // Open the file
    std::ifstream file(filePath);
    if (!file) {
        std::cerr << "Error opening file: " << filePath << std::endl;
        throw std::runtime_error("Error opening file");
    }

    auto start = std::chrono::high_resolution_clock::now();

    std::string line;
    int currentSample = 0;
    std::vector<float> currentLayerStructures;
    std::vector<float> currentCuboidStructures;

    // Clear output vectors
    layerStructures.clear();
    cuboidStructures.clear();

    while (std::getline(file, line)) {
        std::stringstream ss(line);
        std::string group;
        currentLayerStructures.clear();
        currentCuboidStructures.clear();

        while (std::getline(ss, group, ';')) {
            std::stringstream groupStream(group);
            std::vector<float> values;
            float value;
            while (groupStream >> value) {
                values.push_back(value);
                if (groupStream.peek() == ',') {
                    groupStream.ignore();
                }
            }

            if (values.size() == 2) {
                currentLayerStructures.insert(currentLayerStructures.end(), values.begin(), values.end());
            } else if (values.size() == 7) {
                currentCuboidStructures.insert(currentCuboidStructures.end(), values.begin(), values.end());
            }
        }

        // Pad layer structures to MAX_LAYERS*2 elements (each layer has 2 values)
        int currentLayerElements = currentLayerStructures.size();
        int requiredLayerElements = MAX_LAYERS * 2;
        if (currentLayerElements < requiredLayerElements) {
            currentLayerStructures.resize(requiredLayerElements, PADDING_VALUE);
        }

        // Pad cuboid structures to MAX_CUBOIDS*7 elements (each cuboid has 7 values)
        int currentCuboidElements = currentCuboidStructures.size();
        int requiredCuboidElements = MAX_CUBOIDS * 7;
        if (currentCuboidElements < requiredCuboidElements) {
            currentCuboidStructures.resize(requiredCuboidElements, PADDING_VALUE);
        }

        // Append to output vectors
        layerStructures.insert(layerStructures.end(), currentLayerStructures.begin(), currentLayerStructures.end());
        cuboidStructures.insert(cuboidStructures.end(), currentCuboidStructures.begin(), currentCuboidStructures.end());

        currentSample++;

        // Stop reading if we have reached the required number of samples
        if (currentSample >= numSamples) {
            break;
        }
    }

    // Close the file
    file.close();

    auto end = std::chrono::high_resolution_clock::now();
    double totalTime = std::chrono::duration<double>(end - start).count();

    if (verbose) {
        std::cout << "================================================================\nReading input file" << std::endl;
        std::cout << "Total time: " << totalTime << " seconds" << std::endl;
        std::cout << "Total layer elements: " << layerStructures.size() << " (" << currentSample << " samples × " << (MAX_LAYERS * 2) << " elements)" << std::endl;
        std::cout << "Total cuboid elements: " << cuboidStructures.size() << " (" << currentSample << " samples × " << (MAX_CUBOIDS * 7) << " elements)" << std::endl;
        std::cout << "================================================================\n" << std::endl;
    }

    return currentSample;
}

std::pair<torch::Tensor, torch::Tensor> DNNSolver::structureConstruction(
    const std::vector<float>& layerStructures,  const std::vector<float>& cuboidStructures, 
    int numSamples, bool verbose) {
        
    torch::InferenceMode guard;  // Enable inference mode for this method
    
    // Set current stream for all operations
    setCurrentStream();

    // Use pre-allocated tensor slice
    torch::Tensor dielectric = dielectric_buffer_1_.slice(0, 0, numSamples);
    
    auto layerStructuresTensor = torch::from_blob(
        const_cast<float*>(layerStructures.data()),
        {numSamples, MAX_LAYERS, 2},
        torch::TensorOptions().dtype(torch::kFloat32)).to(device);
    
    auto cuboidStructuresTensor = torch::from_blob(
        const_cast<float*>(cuboidStructures.data()),
        {numSamples, MAX_CUBOIDS, 7},
        torch::TensorOptions().dtype(torch::kFloat32)).to(device).clone();
    
    // Extract dielectric values for normalization
    auto layer_dielectrics = layerStructuresTensor.select(2, 1); // [B, L] - position 1 in layers
    auto cuboid_dielectrics = cuboidStructuresTensor.select(2, 6); // [B, C] - position 6 in cuboids
    
    // Find maximum dielectric value per batch
    auto layer_max = layer_dielectrics.amax(1, /*keepdim=*/true); // [B, 1]
    auto cuboid_max = cuboid_dielectrics.amax(1, /*keepdim=*/true); // [B, 1]
    auto global_max = torch::max(layer_max, cuboid_max); // [B, 1]

    // Normalize the dielectric values in-place
    layer_dielectrics.div_(global_max); // Divide by max
    cuboid_dielectrics.div_(global_max);
    
    // Get min_z and normalized values for each layer
    auto min_z_layer = layerStructuresTensor.select(2, 0); // [B, L]
    auto vals_layer = layerStructuresTensor.select(2, 1); // [B, L] - now normalized
    
    // Now directly call your fused CUDA kernel with normalized values
    fill_structure_launcher(
        dielectric, // [B, N, N, N]
        min_z_layer, // [B, L]
        vals_layer, // [B, L] - normalized
        cuboidStructuresTensor // [B, C, 7] - with normalized dielectrics
    );
    
    return std::make_pair(dielectric, global_max);
}