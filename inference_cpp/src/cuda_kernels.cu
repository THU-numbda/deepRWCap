#include <torch/torch.h>
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#define PADDING_VALUE -9999.0f
#define BLOCK_SIZE_3D 8
#define WARP_SIZE 32

#define N 23
#define NN 23*23
#define NNN 23*23*23

#define L 15
#define C 15

// CUDA kernel (no changes needed here)
__global__ void fill_structure_kernel(
    torch::PackedTensorAccessor32<float, 4, torch::RestrictPtrTraits> dielectric,  // [B, N, N, N]
    torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> min_z_layer, // [B, L]
    torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> vals_layer,  // [B, L]
    torch::PackedTensorAccessor32<float, 3, torch::RestrictPtrTraits> cuboids      // [B, C, 7]
    )
{
    int b = blockIdx.x;  // Batch index (sample ID)
    int voxel_idx = threadIdx.x + blockIdx.y * blockDim.x;  // Voxel index within the sample

    int total_voxels = N * N * N;
    if (b >= dielectric.size(0) || voxel_idx >= total_voxels) return;

    // Decode voxel_idx into (x, y, z)
    int x = voxel_idx % N;
    int y = (voxel_idx / N) % N;
    int z = voxel_idx / (N * N);

    // Compute normalized coordinates [-1, 1]
    const float xcoord = (2.0f * (float)x + 1.0f) / (float)N - 1.0f;
    const float ycoord = (2.0f * (float)y + 1.0f) / (float)N - 1.0f;
    const float zcoord = (2.0f * (float)z + 1.0f) / (float)N - 1.0f;

    // -----------------
    // 1. Find layer value
    // -----------------
    int layer_idx = 0;
    for (int l = 0; l < L; ++l) {
        float minz = min_z_layer[b][l];
        if (minz == PADDING_VALUE) break;  // No more valid layers
        if (zcoord >= minz) {
            layer_idx = l;
        } else {
            break;  // Found the correct layer, earlier layers already handled
        }
    }
    float val = vals_layer[b][layer_idx];

    // -----------------
    // 2. Check cuboid overwrite
    // -----------------
    for (int c = 0; c < C; ++c) {
        float minx = cuboids[b][c][0];
        if (minx == PADDING_VALUE) break;  // No cuboid found, exit
        float miny = cuboids[b][c][1];
        float minz = cuboids[b][c][2];
        float maxx = cuboids[b][c][3];
        float maxy = cuboids[b][c][4];
        float maxz = cuboids[b][c][5];
        float cval = cuboids[b][c][6];

        if (xcoord >= minx && xcoord <= maxx &&
            ycoord >= miny && ycoord <= maxy &&
            zcoord >= minz && zcoord <= maxz) {
            val = cval;  // Cuboid overwrites layer
            break;
        }
    }

    // -----------------
    // 3. Write final value
    // -----------------
    dielectric[b][z][y][x] = val;
}

// C++ function to launch the kernel
void launch_fill_structure_kernel(
    torch::Tensor dielectric,
    torch::Tensor min_z_layer,
    torch::Tensor vals_layer,
    torch::Tensor cuboids)
{
    // Get device and set context
    auto device = dielectric.device();
    c10::cuda::set_device(device.index());
    
    // Get the appropriate CUDA stream for this device
    auto stream = c10::cuda::getCurrentCUDAStream(device.index());
    
    const int B = dielectric.size(0);
    const int total_voxels = N * N * N;

    const int threadsPerBlock = 256;
    dim3 threads(threadsPerBlock);
    dim3 blocks(B, (total_voxels + threadsPerBlock - 1) / threadsPerBlock);

    fill_structure_kernel<<<blocks, threads, 0, stream>>>(
        dielectric.packed_accessor32<float, 4, torch::RestrictPtrTraits>(),
        min_z_layer.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
        vals_layer.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
        cuboids.packed_accessor32<float, 3, torch::RestrictPtrTraits>()
    );

    // Check for kernel launch errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error("fill_structure_kernel launch failed: " + 
                                std::string(cudaGetErrorString(err)));
    }
}

__global__ void locate_index_kernel(
    const int16_t* __restrict__ tileIdx,  // [B] - Changed to int16_t
    const uint8_t* __restrict__ faceIdx,  // [B] - uint8_t
    const uint8_t* __restrict__ axis,     // [B] - uint8_t
    float* __restrict__ output,           // [B, 3]
    int B
) {
    int b = blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= B) return;
    
    // Unpack tile ID into x/y
    int16_t idx = tileIdx[b];  // Changed to int16_t
    if (idx < 0 || idx >= NN) return;  // Skip invalid tile indices
    
    int xid = idx % N;
    int yid = idx / N;
    
    // Normalized coordinates
    float temp0 = (2.0f * xid + 1.0f) / N - 1.0f;
    float temp1 = (2.0f * yid + 1.0f) / N - 1.0f;
    float temp2 = 1.0f;
    float temps[3] = {temp0, temp1, temp2};
    
    // Hardcoded face vectors
    const int faceVectors[6][3] = {
        { 1,  2, -3},
        { 1,  2,  3},
        { 1, -3,  2},
        { 1,  3,  2},
        {-3,  1,  2},
        { 3,  1,  2}
    };
    
    uint8_t f = faceIdx[b];
    uint8_t a = axis[b];
    
    // Skip processing if face ID is invalid (255 = no transformation)
    if (f >= 6) return;
    
    // Remap face index based on axis
    if (a == 0) { // X-axis
        if (f == 0) f = 4;
        else if (f == 1) f = 5;
        else if (f == 4) f = 1;
        else if (f == 5) f = 0;
    }
    else if (a == 1) { // Y-axis
        if (f == 0) f = 2;
        else if (f == 1) f = 3;
        else if (f == 2) f = 1;
        else if (f == 3) f = 0;
    }
    
    #pragma unroll
    for (int j = 0; j < 3; ++j) {
        int v = faceVectors[f][j];
        int index = abs(v) - 1;
        int sign = (v < 0) ? -1 : 1;
        output[b * 3 + j] = temps[index] * sign;
    }
}

// Host function for full locate index kernel
void launch_locate_index_kernel(
    torch::Tensor tileIdx, // [B], int16
    torch::Tensor faceIdx, // [B], uint8
    torch::Tensor axis,    // [B], uint8
    torch::Tensor output   // [B, 3], float32
) {
    // Get device and set context
    auto device = tileIdx.device();
    c10::cuda::set_device(device.index());
    
    // Get the appropriate CUDA stream for this device
    auto stream = c10::cuda::getCurrentCUDAStream(device.index());
    
    int B = tileIdx.size(0);
    int threads = 256;
    int blocks = (B + threads - 1) / threads;
    
    locate_index_kernel<<<blocks, threads, 0, stream>>>(
        tileIdx.data_ptr<int16_t>(),   // Changed to int16_t
        faceIdx.data_ptr<uint8_t>(),
        axis.data_ptr<uint8_t>(),
        output.data_ptr<float>(),
        B
    );
    
    // Check for kernel launch errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error("locate_index_kernel launch failed: " +
                                std::string(cudaGetErrorString(err)));
    }
}

// Simplified CUDA kernel - keeps face but no remapping logic
__global__ void locate_index_simple_kernel(
    const int16_t* __restrict__ tileIdx,  // [B] - tile indices
    const uint8_t* __restrict__ faceIdx,  // [B] - face indices (used directly, no remapping)
    float* __restrict__ output,           // [B, 3] - output coordinates
    int B
) {
    int b = blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= B) return;
    
    // Unpack tile ID into x/y
    int16_t idx = tileIdx[b];
    if (idx < 0 || idx >= NN) return;  // Skip invalid tile indices
    
    int xid = idx % N;
    int yid = idx / N;
    
    // Normalized coordinates
    float temp0 = (2.0f * xid + 1.0f) / N - 1.0f;
    float temp1 = (2.0f * yid + 1.0f) / N - 1.0f;
    float temp2 = 1.0f;
    float temps[3] = {temp0, temp1, temp2};
    
    // Hardcoded face vectors
    const int faceVectors[6][3] = {
        { 1,  2, -3},
        { 1,  2,  3},
        { 1, -3,  2},
        { 1,  3,  2},
        {-3,  1,  2},
        { 3,  1,  2}
    };
    
    uint8_t f = faceIdx[b];
    
    // Skip processing if face ID is invalid (255 = no transformation)
    if (f >= 6) return;
    
    // Use face directly without axis remapping
    #pragma unroll
    for (int j = 0; j < 3; ++j) {
        int v = faceVectors[f][j];
        int index = abs(v) - 1;
        int sign = (v < 0) ? -1 : 1;
        output[b * 3 + j] = temps[index] * sign;
    }
}

// Host function for simple locate index kernel
void launch_locate_index_simple_kernel(
    torch::Tensor tileIdx, // [B], int16
    torch::Tensor faceIdx, // [B], uint8
    torch::Tensor output   // [B, 3], float32
) {
    // Get device and set context
    auto device = tileIdx.device();
    c10::cuda::set_device(device.index());
    
    // Get the appropriate CUDA stream for this device
    auto stream = c10::cuda::getCurrentCUDAStream(device.index());
    
    int B = tileIdx.size(0);
    int threads = 256;
    int blocks = (B + threads - 1) / threads;
    
    locate_index_simple_kernel<<<blocks, threads, 0, stream>>>(
        tileIdx.data_ptr<int16_t>(),
        faceIdx.data_ptr<uint8_t>(),
        output.data_ptr<float>(),
        B
    );
    
    // Check for kernel launch errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error("locate_index_simple_kernel launch failed: " +
                                std::string(cudaGetErrorString(err)));
    }
}

// Kernel for Green's function rotations using uint8_t face IDs
__global__ void rotate_faces_poisson_kernel(
    float* __restrict__ output,
    const float* __restrict__ input,
    const uint8_t* __restrict__ face_ids,     // Changed: [batch_size] with values 0-5 (or 255 for no transformation)
    int batch_size) {
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = N * N * N;
    
    if (idx >= batch_size * total_elements) return;
    
    int batch_idx = idx / total_elements;
    int element_idx = idx % total_elements;
    int z = element_idx / (N * N);
    int temp = element_idx % (N * N);
    int y = temp / N;
    int x = temp % N;
    
    // Copy input to output first
    output[idx] = input[idx];
    
    // Get face ID for this batch
    uint8_t face = face_ids[batch_idx];
    
    // Apply transformation based on face ID (255 = no transformation)
    if (face < 6) {
        int src_z, src_y, src_x;
        
        switch(face) {
            case 0:
                return; // No transformation needed
                
            case 1: {
                // For flip(-3): output[z] = input[N-1-z]
                src_z = N - 1 - z;
                src_y = y;
                src_x = x;
                break;
            }
            
            case 2: {
                // rot90(1, {-3, -2}).flip({-3})
                src_z = y;
                src_y = z;
                src_x = x;
                break;
            }
            
            case 3: {
                // rot90(1, {-3, -2})
                src_z = y;
                src_y = N - 1 - z;
                src_x = x;
                break;
            }
            
            case 4: {
                // Complex transformation
                src_z = y;
                src_y = x;
                src_x = z;
                break;
            }
            
            case 5: {
                // Complex transformation  
                src_z = y;
                src_y = x;
                src_x = N - 1 - z;
                break;
            }
        }
        
        // Calculate source index
        int src_idx = batch_idx * total_elements + src_z * N * N + src_y * N + src_x;
        
        // Bounds check
        if (src_z >= 0 && src_z < N && src_y >= 0 && src_y < N && src_x >= 0 && src_x < N) {
            output[idx] = input[src_idx];
        }
    }
}

// NEW: Buffer-based launcher for Green's function rotation
void launch_rotate_faces_poisson_with_buffer(
    torch::Tensor& output,     // Pre-allocated output buffer
    torch::Tensor& input,      // Input buffer
    torch::Tensor& face_ids) {
    
    int B = input.size(0);
    int total_elements = B * N * N * N;
    
    float* input_ptr = input.data_ptr<float>();
    float* output_ptr = output.data_ptr<float>();
    uint8_t* face_ptr = face_ids.data_ptr<uint8_t>();
    
    int threads_per_block = 256;
    int blocks = (total_elements + threads_per_block - 1) / threads_per_block;
    
    auto stream = c10::cuda::getCurrentCUDAStream(input.device().index());
    
    rotate_faces_poisson_kernel<<<blocks, threads_per_block, 0, stream>>>(
        output_ptr, input_ptr, face_ptr, B);
    
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error("rotate_faces_poisson_kernel failed: " + 
                                std::string(cudaGetErrorString(err)));
    }
}

// Kernel for gradient/structure movement rotations using uint8_t face IDs
__global__ void rotate_faces_gradient_kernel(
    float* __restrict__ output,
    const float* __restrict__ input,
    const uint8_t* __restrict__ face_ids,     // Changed: [batch_size] with values 0-5 (or 255 for no transformation)
    int batch_size) {
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = N * N * N;
    
    if (idx >= batch_size * total_elements) return;
    
    int batch_idx = idx / total_elements;
    int element_idx = idx % total_elements;
    int z = element_idx / (N * N);
    int temp = element_idx % (N * N);
    int y = temp / N;
    int x = temp % N;
    
    // Copy input to output first
    output[idx] = input[idx];
    
    // Get face ID for this batch
    uint8_t face = face_ids[batch_idx];
    
    // Apply transformation based on face ID (255 = no transformation)
    if (face < 6) {
        int src_z, src_y, src_x;
        
        switch(face) {
            case 0: {
                // flip({-3}): output[z] = input[N-1-z]
                src_z = N - 1 - z;
                src_y = y;
                src_x = x;
                break;
            }
            case 1:
            case 2:
                // No transformation for faces 1 and 2
                return;
                
            case 3: {
                // flip({-2}): output[y] = input[N-1-y]
                src_z = z;
                src_y = N - 1 - y;
                src_x = x;
                break;
            }
            case 4: {
                // rot90(1, {-2,-1}).flip({-2})
                // First rot90: (y,x) -> (N-1-x, y)
                // Then flip({-2}): y -> N-1-y
                // Combined: input[z,y,x] -> output[z, N-1-(N-1-x), y] = output[z, x, y]
                src_z = z;
                src_y = x;
                src_x = y;
                break;
            }
            case 5: {
                // rot90(1, {-2,-1}): (y,x) -> (N-1-x, y)
                src_z = z;
                src_y = x;
                src_x =  N - 1 - y;
                break;
            }
        }
        
        // Calculate source index
        int src_idx = batch_idx * total_elements + src_z * N * N + src_y * N + src_x;
        
        // Bounds check
        if (src_z >= 0 && src_z < N && src_y >= 0 && src_y < N && src_x >= 0 && src_x < N) {
            output[idx] = input[src_idx];
        }
    }
}

// NEW: Buffer-based launcher for gradient rotation
void launch_rotate_faces_gradient_with_buffer(
    torch::Tensor& output,     // Pre-allocated output buffer
    torch::Tensor& input,      // Input buffer
    torch::Tensor& face_ids) {
    
    int B = input.size(0);
    int total_elements = B * N * N * N;
    
    float* input_ptr = input.data_ptr<float>();
    float* output_ptr = output.data_ptr<float>();
    uint8_t* face_ptr = face_ids.data_ptr<uint8_t>();
    
    int threads_per_block = 256;
    int blocks = (total_elements + threads_per_block - 1) / threads_per_block;
    
    auto stream = c10::cuda::getCurrentCUDAStream(input.device().index());
    
    rotate_faces_gradient_kernel<<<blocks, threads_per_block, 0, stream>>>(
        output_ptr, input_ptr, face_ptr, B);
    
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error("rotate_faces_gradient_kernel failed: " + 
                                std::string(cudaGetErrorString(err)));
    }
}

__global__ void post_process_gradient_kernel(
    float* __restrict__ input,        // Shape: [batch_size, N*N] - 2D flattened
    float* __restrict__ output,        // Shape: [batch_size, N*N] - 2D flattened
    const uint8_t* __restrict__ face_ids,    // Changed: [batch_size] with values 0-5 (or 255 for no transformation)
    const uint8_t* __restrict__ axis,        // Changed: [batch_size] with values 0-2
    int batch_size) {
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = N * N;  // 2D surface, not 3D volume
    
    if (idx >= batch_size * total_elements) return;
    
    int batch_idx = idx / total_elements;
    int element_idx = idx % total_elements;
    int y = element_idx / N;  // No Z dimension - it's a 2D face
    int x = element_idx % N;
    
    // Get face ID and axis for this batch
    uint8_t face = face_ids[batch_idx];
    uint8_t ax = axis[batch_idx];
    bool axX = (ax == 0);
    bool axY = (ax == 1);
    
    // Determine face groups based on face ID
    bool face01 = (face == 0 || face == 1);
    bool face45 = (face == 4 || face == 5);
    
    // 4.2) Combined rotation and flip logic
    bool should_rotate = axX || (axY && face45);
    bool should_flip = (axX && face45) || (axY && face01);
    
    if (should_rotate || should_flip) {
        int src_y = y, src_x = x;

        if (should_flip && should_rotate) {
            // (rot90(1, {-2,-1}.flip({-2}))
            src_y = x;
            src_x = y;
        } else if (should_rotate) {
            // (rot90(1, {-2,-1}) For 2D: (y,x) -> (N-1-x, y)
            src_y = x;
            src_x = N - 1 - y;
        } else if (should_flip) {
            src_y = N - 1 - y;
        }
        
        // Read from source position
        int src_idx = batch_idx * total_elements + src_y * N + src_x;
        
        // Final assignment with bounds check
        if (src_y >= 0 && src_y < N && src_x >= 0 && src_x < N) {
            output[idx] = input[src_idx];
        }
    } else {
        // No transformation needed, just copy
        output[idx] = input[idx];
    }

    // 4.1) Negate face 0
    if (face == 0) {
        output[idx] = -output[idx];
    }
}

// NEW: Buffer-based launcher for post-processing gradient
void launch_post_process_gradient_with_buffer(
    torch::Tensor& output,     // Pre-allocated output buffer
    torch::Tensor& input,      // Input gradient
    torch::Tensor& face_ids,
    torch::Tensor& axis) {
    
    int B = input.size(0);
    
    float* input_ptr = input.data_ptr<float>();
    float* output_ptr = output.data_ptr<float>();
    uint8_t* face_ptr = face_ids.data_ptr<uint8_t>();
    uint8_t* axis_ptr = axis.data_ptr<uint8_t>();
    
    int threads_per_block = 256;
    int blocks = (B * NN + threads_per_block - 1) / threads_per_block;
    
    auto stream = c10::cuda::getCurrentCUDAStream(input.device().index());
    
    post_process_gradient_kernel<<<blocks, threads_per_block, 0, stream>>>(
        input_ptr, output_ptr, face_ptr, axis_ptr, B);
    
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error("post_process_gradient_kernel failed: " + 
                                std::string(cudaGetErrorString(err)));
    }
}