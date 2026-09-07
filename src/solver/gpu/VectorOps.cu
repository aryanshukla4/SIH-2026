#include "sovsolve/solver/gpu/VectorOps.hpp"

#include <string>

#include <cuda_runtime.h>

namespace sovsolve::solver::gpu {

namespace {

constexpr int kBlockSize = 256;

__global__ void hadamard_kernel(const Real* a, const Real* b, Real* y, std::size_t n) {
  const std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < n) y[i] = a[i] * b[i];
}

__global__ void hadamard_add_kernel(const Real* a, const Real* b, Real* y, std::size_t n) {
  const std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < n) y[i] += a[i] * b[i];
}

Status launch_check(const char* what) {
  const cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    return core::make_error(core::ErrorCode::NumericalError,
                            std::string(what) + " launch failed: " + cudaGetErrorString(err));
  }
  return Status::Ok();
}

}  // namespace

Status hadamard(const Real* a, const Real* b, Real* y, std::size_t n) {
  if (n == 0) return Status::Ok();
  const int blocks = static_cast<int>((n + kBlockSize - 1) / kBlockSize);
  hadamard_kernel<<<blocks, kBlockSize>>>(a, b, y, n);
  return launch_check("hadamard");
}

Status hadamard_add(const Real* a, const Real* b, Real* y, std::size_t n) {
  if (n == 0) return Status::Ok();
  const int blocks = static_cast<int>((n + kBlockSize - 1) / kBlockSize);
  hadamard_add_kernel<<<blocks, kBlockSize>>>(a, b, y, n);
  return launch_check("hadamard_add");
}

}  // namespace sovsolve::solver::gpu
