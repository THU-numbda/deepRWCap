#include "cuda_kernels.h"
#include <c10/cuda/CUDAGuard.h>

// Forward declarations
void launch_fill_structure_kernel(
   torch::Tensor dielectric,
   torch::Tensor min_z_layer,
   torch::Tensor vals_layer,
   torch::Tensor cuboids);

void launch_locate_index_kernel(
   torch::Tensor tileIdx,
   torch::Tensor faceIdx,
   torch::Tensor axis,
   torch::Tensor output);

void launch_locate_index_simple_kernel(
   torch::Tensor tileIdx,
   torch::Tensor faceIdx,
   torch::Tensor output);

void launch_rotate_faces_greens_with_buffer(
   torch::Tensor& output,
   torch::Tensor& input,
   torch::Tensor& face_ids);

void launch_rotate_faces_gradient_with_buffer(
   torch::Tensor& output,
   torch::Tensor& input,
   torch::Tensor& face_ids);

void launch_post_process_gradient_with_buffer(
   torch::Tensor& output,
   torch::Tensor& input,
   torch::Tensor& face_ids,
   torch::Tensor& axis);

void fill_structure_launcher(
   torch::Tensor dielectric,
   torch::Tensor min_z_layer,
   torch::Tensor vals_layer,
   torch::Tensor cuboids)
{
   TORCH_CHECK(dielectric.is_cuda(), "dielectric must be a CUDA tensor");
   TORCH_CHECK(min_z_layer.is_cuda(), "min_z_layer must be a CUDA tensor");
   TORCH_CHECK(vals_layer.is_cuda(), "vals_layer must be a CUDA tensor");
   TORCH_CHECK(cuboids.is_cuda(), "cuboids must be a CUDA tensor");
   
   // Ensure all tensors are on the same device
   auto device = dielectric.device();
   TORCH_CHECK(min_z_layer.device() == device, "All tensors must be on the same device");
   TORCH_CHECK(vals_layer.device() == device, "All tensors must be on the same device");
   TORCH_CHECK(cuboids.device() == device, "All tensors must be on the same device");
   
   // Set device context for the kernel launch
   c10::cuda::set_device(device.index());
   launch_fill_structure_kernel(dielectric, min_z_layer, vals_layer, cuboids);
}

void locate_index_launcher(
   torch::Tensor tileIdx,
   torch::Tensor faceIdx,
   torch::Tensor axis,
   torch::Tensor output)
{
   TORCH_CHECK(tileIdx.is_cuda(), "tileIdx must be CUDA");
   TORCH_CHECK(faceIdx.is_cuda(), "faceIdx must be CUDA");
   TORCH_CHECK(axis.is_cuda(), "axis must be CUDA");
   TORCH_CHECK(output.is_cuda(), "output must be CUDA");
   
   // Check data types
   TORCH_CHECK(tileIdx.dtype() == torch::kInt16, "tileIdx must be int16");
   TORCH_CHECK(faceIdx.dtype() == torch::kUInt8, "faceIdx must be uint8");
   TORCH_CHECK(axis.dtype() == torch::kUInt8, "axis must be uint8");
   TORCH_CHECK(output.dtype() == torch::kFloat32, "output must be float32");
   
   // Ensure all tensors are on the same device
   auto device = tileIdx.device();
   TORCH_CHECK(faceIdx.device() == device, "All tensors must be on the same device");
   TORCH_CHECK(axis.device() == device, "All tensors must be on the same device");
   TORCH_CHECK(output.device() == device, "All tensors must be on the same device");
   
   // Set device context for the kernel launch
   c10::cuda::set_device(device.index());
   launch_locate_index_kernel(tileIdx, faceIdx, axis, output);
}

// Simple locate index launcher implementation
void locate_index_simple_launcher(
   torch::Tensor tileIdx,
   torch::Tensor faceIdx,
   torch::Tensor output)
{
   TORCH_CHECK(tileIdx.is_cuda(), "tileIdx must be CUDA");
   TORCH_CHECK(faceIdx.is_cuda(), "faceIdx must be CUDA");
   TORCH_CHECK(output.is_cuda(), "output must be CUDA");
   
   // Check data types
   TORCH_CHECK(tileIdx.dtype() == torch::kInt16, "tileIdx must be int16");
   TORCH_CHECK(faceIdx.dtype() == torch::kUInt8, "faceIdx must be uint8");
   TORCH_CHECK(output.dtype() == torch::kFloat32, "output must be float32");
   
   // Ensure tensors are on the same device
   auto device = tileIdx.device();
   TORCH_CHECK(faceIdx.device() == device, "All tensors must be on the same device");
   TORCH_CHECK(output.device() == device, "All tensors must be on the same device");
   
   // Set device context for the kernel launch
   c10::cuda::set_device(device.index());
   launch_locate_index_simple_kernel(tileIdx, faceIdx, output);
}

// Green's function rotation with pre-allocated buffer (returns slice)
torch::Tensor rotate_faces_greens_launcher(
   torch::Tensor& output_buffer,
   torch::Tensor& input,
   torch::Tensor& face_ids) {
   
   TORCH_CHECK(input.is_cuda(), "input must be CUDA");
   TORCH_CHECK(output_buffer.is_cuda(), "output_buffer must be CUDA");
   TORCH_CHECK(face_ids.is_cuda(), "face_ids must be CUDA");
   TORCH_CHECK(input.dtype() == torch::kFloat32, "input must be float32");
   TORCH_CHECK(output_buffer.dtype() == torch::kFloat32, "output_buffer must be float32");
   TORCH_CHECK(face_ids.dtype() == torch::kUInt8, "face_ids must be uint8");
   
   auto device = input.device();
   TORCH_CHECK(output_buffer.device() == device, "All tensors must be on the same device");
   TORCH_CHECK(face_ids.device() == device, "All tensors must be on the same device");
   
   int batch_size = input.size(0);
   TORCH_CHECK(batch_size <= output_buffer.size(0), "Output buffer too small for batch size");
   
   // Get the slice we need
   auto output_slice = output_buffer.slice(0, 0, batch_size);
   
   c10::cuda::set_device(device.index());
   launch_rotate_faces_greens_with_buffer(output_slice, input, face_ids);
   
   return output_slice;
}

// Gradient rotation with pre-allocated buffer (returns slice)
torch::Tensor rotate_faces_gradient_launcher(
   torch::Tensor& output_buffer,
   torch::Tensor& input,
   torch::Tensor& face_ids) {
   
   TORCH_CHECK(input.is_cuda(), "input must be CUDA");
   TORCH_CHECK(output_buffer.is_cuda(), "output_buffer must be CUDA");
   TORCH_CHECK(face_ids.is_cuda(), "face_ids must be CUDA");
   TORCH_CHECK(input.dtype() == torch::kFloat32, "input must be float32");
   TORCH_CHECK(output_buffer.dtype() == torch::kFloat32, "output_buffer must be float32");
   TORCH_CHECK(face_ids.dtype() == torch::kUInt8, "face_ids must be uint8");
   
   auto device = input.device();
   TORCH_CHECK(output_buffer.device() == device, "All tensors must be on the same device");
   TORCH_CHECK(face_ids.device() == device, "All tensors must be on the same device");
   
   int batch_size = input.size(0);
   TORCH_CHECK(batch_size <= output_buffer.size(0), "Output buffer too small for batch size");
   
   // Get the slice we need
   auto output_slice = output_buffer.slice(0, 0, batch_size);
   
   c10::cuda::set_device(device.index());
   launch_rotate_faces_gradient_with_buffer(output_slice, input, face_ids);
   
   return output_slice;
}

// NEW: Post-process gradient with pre-allocated buffer (returns slice)
torch::Tensor post_process_gradient_launcher(
   torch::Tensor& output_buffer,
   torch::Tensor& input_gradient,
   torch::Tensor& face_ids,
   torch::Tensor& axis) {
   
   TORCH_CHECK(input_gradient.is_cuda(), "input_gradient must be CUDA");
   TORCH_CHECK(output_buffer.is_cuda(), "output_buffer must be CUDA");
   TORCH_CHECK(face_ids.is_cuda(), "face_ids must be CUDA");
   TORCH_CHECK(axis.is_cuda(), "axis must be CUDA");
   TORCH_CHECK(input_gradient.dtype() == torch::kFloat32, "input_gradient must be float32");
   TORCH_CHECK(output_buffer.dtype() == torch::kFloat32, "output_buffer must be float32");
   TORCH_CHECK(face_ids.dtype() == torch::kUInt8, "face_ids must be uint8");
   TORCH_CHECK(axis.dtype() == torch::kUInt8, "axis must be uint8");
   
   auto device = input_gradient.device();
   TORCH_CHECK(output_buffer.device() == device, "All tensors must be on the same device");
   TORCH_CHECK(face_ids.device() == device, "All tensors must be on the same device");
   TORCH_CHECK(axis.device() == device, "All tensors must be on the same device");
   
   int batch_size = input_gradient.size(0);
   TORCH_CHECK(batch_size <= output_buffer.size(0), "Output buffer too small for batch size");
   
   // Get the slice we need
   auto output_slice = output_buffer.slice(0, 0, batch_size);
   
   c10::cuda::set_device(device.index());
   launch_post_process_gradient_with_buffer(output_slice, input_gradient, face_ids, axis);
   
   return output_slice;
}

// Convenience function that creates and returns output tensor (full version)
torch::Tensor locate_index_cuda_launcher(
   torch::Tensor tileIdx,
   torch::Tensor faceIdx,
   torch::Tensor axis)
{
   TORCH_CHECK(tileIdx.is_cuda(), "tileIdx must be CUDA");
   TORCH_CHECK(faceIdx.is_cuda(), "faceIdx must be CUDA");
   TORCH_CHECK(axis.is_cuda(), "axis must be CUDA");
   
   // Check data types
   TORCH_CHECK(tileIdx.dtype() == torch::kInt16, "tileIdx must be int16");
   TORCH_CHECK(faceIdx.dtype() == torch::kUInt8, "faceIdx must be uint8");
   TORCH_CHECK(axis.dtype() == torch::kUInt8, "axis must be uint8");
   
   int B = tileIdx.size(0);
   auto output = torch::zeros({B, 3}, torch::dtype(torch::kFloat32).device(tileIdx.device()));
   
   locate_index_launcher(tileIdx, faceIdx, axis, output);
   return output;
}

// Convenience function that creates and returns output tensor (simple version)
torch::Tensor locate_index_simple_cuda_launcher(
   torch::Tensor tileIdx,
   torch::Tensor faceIdx)
{
   TORCH_CHECK(tileIdx.is_cuda(), "tileIdx must be CUDA");
   TORCH_CHECK(faceIdx.is_cuda(), "faceIdx must be CUDA");
   TORCH_CHECK(tileIdx.dtype() == torch::kInt16, "tileIdx must be int16");
   TORCH_CHECK(faceIdx.dtype() == torch::kUInt8, "faceIdx must be uint8");
   
   int B = tileIdx.size(0);
   auto output = torch::zeros({B, 3}, torch::dtype(torch::kFloat32).device(tileIdx.device()));
   
   locate_index_simple_launcher(tileIdx, faceIdx, output);
   return output;
}