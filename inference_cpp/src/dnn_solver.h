#pragma once

#include <torch/script.h>
#include <torch/torch.h>
#include <string>
#include <vector>
#include <functional>
#include <mutex>
#include <condition_variable>
#include <atomic>
#include <stdexcept>
#include <future>
#include <omp.h>
#include <thread>
#include <concurrentqueue.h>
#include <optional>  // Add this include

// CUDA headers for stream management
#include <c10/cuda/CUDAStream.h>

// Only load once in the entire application
#include <torch_tensorrt/torch_tensorrt.h>

#define MODELS_PATH "models.txt"

// Structure related parameters
const int MAX_SAMPLES = 2048;
const int MAX_CUBOIDS = 15;
const int MAX_LAYERS = 15;
const int PADDING_VALUE = -9999.0f;

enum class Axis : int {
    X = 0,
    Y = 1,
    Z = 2
};

enum class DataType {
    FLOAT,
    DOUBLE
};

class DNNSolver {
public:
    static constexpr int N = 23;
    static constexpr int NN = N*N;
    static constexpr int NNN = N*N*N;

    // Constructor
    DNNSolver(int device_idx = 0);

    // Static utility functions (common to all solvers)
    static void readBinaryFile(const std::string& filePath, std::vector<float>& dielectricConfigs, 
                              std::vector<float>& targetFunctions, int numSamples, int structureSize, 
                              DataType inputType, bool verbose);
    
    int readStructureFile(const std::string& filePath, std::vector<float>& layerStructures,
                         std::vector<float>& cuboidStructures, int numSamples, bool verbose);
    
    std::pair<torch::Tensor, torch::Tensor> structureConstruction(
                        const std::vector<float>& layerStructures, const std::vector<float>& cuboidStructures, 
                        int numSamples, bool verbose);

    // Pure virtual functions (must be implemented by derived classes)
    virtual void loadModels(bool verbose = false) = 0;
    
    // Virtual sample methods - different signatures for different solver types
    virtual std::vector<float> sample(const std::vector<float>& layerStructures, 
                                     const std::vector<float>& cuboidStructures) = 0;
    
    virtual std::vector<float> sample(const std::vector<float>& layerStructures, 
                                     const std::vector<float>& cuboidStructures, 
                                     const std::vector<int>& axis) = 0;

protected:
    // Protected members (accessible by derived classes)
    torch::Device device;

    // CUDA stream for non-default stream operations - using optional to avoid initialization issues
    std::optional<c10::cuda::CUDAStream> cuda_stream_;

    // Pre-allocated working buffers
    torch::Tensor dielectric_buffer_1_;  // Primary buffer
    torch::Tensor dielectric_buffer_2_;  // Secondary buffer for ping-pong
    torch::Tensor gradient_buffer_1_;    // For gradient operations
    torch::Tensor gradient_buffer_2_;    // Secondary gradient buffer

    // Protected helper functions
    torch::jit::script::Module loadModel(const std::string& modelPath, bool verbose = false);
    std::vector<std::string> loadModelPaths(const std::string& filename);
    void setCurrentStream();  // Helper to set current CUDA stream

    // Friend class for testing
    friend class DNNSolverTests;
};

// Rest of the classes remain the same...
class DNNSolverPoisson : public DNNSolver {
public:
    // Constructor
    DNNSolverPoisson(int device_idx = 0);

    // Override virtual functions
    void loadModels(bool verbose = false) override;
    
    // Green's function sample methods
    std::vector<float> sample(const std::vector<float>& layerStructures, 
                             const std::vector<float>& cuboidStructures) override;
    
    // For Green's function, axis parameter is ignored
    std::vector<float> sample(const std::vector<float>& layerStructures, 
                             const std::vector<float>& cuboidStructures, 
                             const std::vector<int>& axis) override;

private:
    // Green's function specific models
    torch::jit::script::Module faceSelector;
    torch::jit::script::Module greenPredictor;

    // Green's function specific inference functions
    torch::Tensor samplePoissonFunction(torch::Tensor& dielectricTensor);
};

class DNNSolverGrad : public DNNSolver {
public:
    // Constructor
    DNNSolverGrad(int device_idx = 0);

    // Override virtual functions
    void loadModels(bool verbose = false) override;
    
    // Gradient sample methods
    std::vector<float> sample(const std::vector<float>& layerStructures, 
                             const std::vector<float>& cuboidStructures) override;
    
    std::vector<float> sample(const std::vector<float>& layerStructures, 
                             const std::vector<float>& cuboidStructures, 
                             const std::vector<int>& axis) override;

private:
    // Gradient specific models
    torch::jit::script::Module gradientFaceSelectorWeightPredictor;
    torch::jit::script::Module gradientFace2PredictorWithSign;
    torch::jit::script::Module gradientFace1Predictor;

    // Gradient specific inference functions and utilities
    torch::Tensor sampleGradient(torch::Tensor& dielectricTensor, torch::Tensor& max_vals, torch::Tensor& axis);
    torch::Tensor predictGradient(torch::Tensor& dielectricTensor, torch::Tensor& faceIds, torch::Tensor& axis);
};

// ParallelSampler class remains the same...
class ParallelSampler {
public:
    // Constructor takes a solver pointer
    ParallelSampler(DNNSolver* solver, size_t sample_size, size_t batch_size);
    // Destructor
    ~ParallelSampler();
    // Submit methods
    std::future<std::vector<float>> submit(const std::vector<float>& layer, const std::vector<float>& cuboid);
    std::future<std::vector<float>> submit(const std::vector<float>& layer, const std::vector<float>& cuboid, const int axis);

private:
    // Task structure
    struct Task {
        std::vector<float> input_a; // Layer structures
        std::vector<float> input_b; // Cuboid structures
        std::vector<int> axis; // Axis parameter (empty if not needed)
        std::shared_ptr<std::promise<std::vector<float>>> promise; // Promise to fulfill
    };
    
    // Private methods
    void gpuWorkerLoop(); // Main worker loop for GPU thread
    void processBatch(std::vector<std::shared_ptr<Task>>& batch); // Process a batch of tasks
    
    // Solver pointer
    DNNSolver* solver_;
    
    // Config parameters
    const size_t sample_size_; // Size of each sample
    const size_t batch_size_; // Size of a batch
    
    // Lock-free task queue - this replaces the old queue + mutex
    moodycamel::ConcurrentQueue<std::shared_ptr<Task>> task_queue_;
    
    // Minimal synchronization for worker thread sleep/wake
    std::mutex cv_mutex_;
    std::condition_variable queue_cv_;
    
    // Processing state
    std::thread gpu_thread_; // Thread for GPU processing
    std::atomic<bool> shutdown_{false}; // Signal for shutdown
    
    // Pre-allocated buffer for bulk dequeue operations
    static constexpr size_t MAX_BATCH_SIZE = MAX_SAMPLES;
    std::shared_ptr<Task> batch_buffer_[MAX_BATCH_SIZE];
    
    // Pre-allocated reusable buffers to avoid repeated allocations
    std::vector<float> input_a_buffer_;
    std::vector<float> input_b_buffer_;
    std::vector<int> axes_buffer_;
    std::vector<float> result_buffer_;
};