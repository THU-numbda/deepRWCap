#pragma once
#include <torch/torch.h>
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>

// Fill structure kernel launcher
void fill_structure_launcher(
    torch::Tensor dielectric,
    torch::Tensor min_z_layer,
    torch::Tensor vals_layer,
    torch::Tensor cuboids);

// Locate index kernel launcher (int16 + uint8 version)
void locate_index_launcher(
    torch::Tensor tileIdx,    // [B], int16
    torch::Tensor faceIdx,    // [B], uint8
    torch::Tensor axis,       // [B], uint8
    torch::Tensor output);    // [B, 3], float32

// Simple locate index kernel launcher (int16 + uint8 - keeps face but no remapping)
void locate_index_simple_launcher(
    torch::Tensor tileIdx,    // [B], int16
    torch::Tensor faceIdx,    // [B], uint8
    torch::Tensor output);    // [B, 3], float32

// Face rotation kernels for Green's function (buffer-based, returns slice)
torch::Tensor rotate_faces_greens_launcher(
    torch::Tensor& output_buffer,    // Pre-allocated output buffer
    torch::Tensor& input,            // Input tensor
    torch::Tensor& face_ids);        // [batch_size], uint8

// Face rotation kernels for gradient/structure movement (buffer-based, returns slice)
torch::Tensor rotate_faces_gradient_launcher(
    torch::Tensor& output_buffer,    // Pre-allocated output buffer
    torch::Tensor& input,            // Input tensor
    torch::Tensor& face_ids);        // [batch_size], uint8

// Post-processing gradient kernel (buffer-based, returns slice)
torch::Tensor post_process_gradient_launcher(
    torch::Tensor& output_buffer,    // Pre-allocated output buffer
    torch::Tensor& input_gradient,   // Input gradient
    torch::Tensor& face_ids,         // [batch_size], uint8
    torch::Tensor& axis);            // [batch_size], uint8

// Convenience function that returns output tensor (full version)
torch::Tensor locate_index_cuda_launcher(
    torch::Tensor tileIdx,           // [B], int16
    torch::Tensor faceIdx,           // [B], uint8
    torch::Tensor axis);             // [B], uint8

// Convenience function that returns output tensor (simple version)
torch::Tensor locate_index_simple_cuda_launcher(
    torch::Tensor tileIdx,           // [B], int16
    torch::Tensor faceIdx);          // [B], uint8