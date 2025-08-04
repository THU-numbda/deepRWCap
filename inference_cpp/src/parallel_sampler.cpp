#include "dnn_solver.h"
#include <stdexcept>
#include <iostream>
#include <algorithm>
#include <chrono>
#include <thread>

// Static counter to distribute GPU threads across available cores
static std::atomic<int> gpu_thread_counter{0};

// Constructor takes a solver pointer
ParallelSampler::ParallelSampler(DNNSolver* solver, size_t sample_size, size_t batch_size)
    : solver_(solver), sample_size_(sample_size), batch_size_(batch_size) {
    
    if (!solver_) {
        throw std::invalid_argument("Solver pointer cannot be null");
    }
    
    gpu_thread_ = std::thread([this]() { 
        gpuWorkerLoop(); 
    });
}

// Destructor
ParallelSampler::~ParallelSampler() {
    // Signal shutdown
    shutdown_ = true;
    
    // Wake up the worker thread
    {
        std::lock_guard<std::mutex> lock(cv_mutex_);
    }
    queue_cv_.notify_all();
    
    // Join the GPU thread
    if (gpu_thread_.joinable()) {
        gpu_thread_.join();
    }
}

// Submit method without axis
std::future<std::vector<float>> ParallelSampler::submit(const std::vector<float>& layer, const std::vector<float>& cuboid) {
    auto task = std::make_shared<Task>();
    task->input_a = layer;
    task->input_b = cuboid;
    task->axis.clear(); // No axis
    task->promise = std::make_shared<std::promise<std::vector<float>>>();
    
    auto future = task->promise->get_future();
    
    // Lock-free enqueue!
    task_queue_.enqueue(task);
    
    // Wake up worker if it's sleeping (minimal locking)
    {
        std::lock_guard<std::mutex> lock(cv_mutex_);
    }
    queue_cv_.notify_one();
    
    return future;
}

// Submit method with axis
std::future<std::vector<float>> ParallelSampler::submit(const std::vector<float>& layer, const std::vector<float>& cuboid, const int axis) {
    auto task = std::make_shared<Task>();
    task->input_a = layer;
    task->input_b = cuboid;
    task->axis = {axis}; // Single axis value
    task->promise = std::make_shared<std::promise<std::vector<float>>>();
    
    auto future = task->promise->get_future();
    
    // Lock-free enqueue!
    task_queue_.enqueue(task);
    
    // Wake up worker if it's sleeping
    {
        std::lock_guard<std::mutex> lock(cv_mutex_);
    }
    queue_cv_.notify_one();
    
    return future;
}

void ParallelSampler::gpuWorkerLoop() {
    while (!shutdown_) {
        size_t dequeued = task_queue_.try_dequeue_bulk(batch_buffer_,
                                                      std::min(batch_size_, MAX_BATCH_SIZE));
        
        if (dequeued > 0) {
            std::vector<std::shared_ptr<Task>> batch(batch_buffer_, batch_buffer_ + dequeued);
            
            try {
                processBatch(batch);
            } catch (const std::exception& e) {
                for (auto& task : batch) {
                    try {
                        task->promise->set_exception(std::current_exception());
                    } catch (...) {}
                }
            }
        } else {
            // Block indefinitely until work arrives - no polling!
            std::unique_lock<std::mutex> lock(cv_mutex_);
            queue_cv_.wait(lock, [this] {
                return shutdown_ || task_queue_.size_approx() > 0;
            });
        }
    }
    
    // Process any remaining tasks during shutdown
    size_t remaining;
    while ((remaining = task_queue_.try_dequeue_bulk(batch_buffer_, MAX_BATCH_SIZE)) > 0) {
        std::vector<std::shared_ptr<Task>> batch(batch_buffer_, batch_buffer_ + remaining);
        try {
            processBatch(batch);
        } catch (const std::exception& e) {
            for (auto& task : batch) {
                try {
                    task->promise->set_exception(std::current_exception());
                } catch (...) {}
            }
        }
    }
}
// Process a batch of tasks using the solver's sample methods
void ParallelSampler::processBatch(std::vector<std::shared_ptr<Task>>& batch) {
    if (batch.empty()) return;

    // Prepare batched inputs using static thread_local caches
    static thread_local std::vector<float> input_a_cache;
    static thread_local std::vector<float> input_b_cache;
    static thread_local std::vector<int> axes_cache;

    // Clear caches for reuse
    input_a_cache.clear();
    input_b_cache.clear();
    axes_cache.clear();

    // Calculate total input size
    size_t total_size_a = 0;
    size_t total_size_b = 0;
    bool has_axis = false;

    for (const auto& task : batch) {
        total_size_a += task->input_a.size();
        total_size_b += task->input_b.size();
        if (!task->axis.empty()) {
            has_axis = true;
        }
    }

    // Reserve space for efficiency using caches
    input_a_cache.reserve(total_size_a);
    input_b_cache.reserve(total_size_b);
    if (has_axis) {
        axes_cache.reserve(batch.size());
    }

    // Collect all inputs using caches
    for (const auto& task : batch) {
        input_a_cache.insert(input_a_cache.end(), task->input_a.begin(), task->input_a.end());
        input_b_cache.insert(input_b_cache.end(), task->input_b.begin(), task->input_b.end());
        
        if (has_axis) {
            if (!task->axis.empty()) {
                axes_cache.push_back(task->axis[0]); // Take first axis value
            } else {
                axes_cache.push_back(0); // Default axis if not provided
            }
        }
    }

    // Process the batch using the solver's sample method
    std::vector<float> result;
    try {
        if (has_axis) {
            // Call the version with axis using caches
            result = solver_->sample(input_a_cache, input_b_cache, axes_cache);
        } else {
            // Call the version without axis using caches
            result = solver_->sample(input_a_cache, input_b_cache);
        }
    } catch (const std::exception& e) {
        // If the solver throws an exception, propagate to all tasks
        for (auto& task : batch) {
            try {
                task->promise->set_exception(std::current_exception());
            } catch (...) {}
        }
        return;
    }

    // Distribute results back to individual tasks
    if (result.size() % batch.size() != 0) {
        // Handle error case where result size doesn't divide evenly
        std::runtime_error error("Result size doesn't match expected batch output size");
        for (auto& task : batch) {
            try {
                task->promise->set_exception(std::make_exception_ptr(error));
            } catch (...) {}
        }
        return;
    }

    size_t result_size = result.size() / batch.size();
    for (size_t i = 0; i < batch.size(); ++i) {
        std::vector<float> task_result(result.begin() + i * result_size,
                                      result.begin() + (i + 1) * result_size);
        batch[i]->promise->set_value(std::move(task_result));
    }
}