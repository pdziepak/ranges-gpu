/*
 * Copyright © 2019 Paweł Dziepak
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */

#pragma once

#include <cassert>

#include <vector>

#include "ranges-gpu/array.hpp"
#include "ranges-gpu/span.hpp"
#include "ranges-gpu/view/detail.hpp"

#include "detail.hpp"

namespace ranges_gpu {
namespace action {

namespace detail {

template<typename V> __global__ auto materialize(size_t total_size, V in, typename V::value_type* out) -> void {
  auto id = blockIdx.x * blockDim.x + threadIdx.x;
  if (id < total_size) { out[id] = in[id]; }
}

template<typename V> auto to_gpu_fn_impl(known_size_tag<true>, V in) -> array<typename V::value_type> {
  using value_type = typename V::value_type;
  static constexpr size_t max_block_size = 1024;

  size_t total_size = in.size();
  size_t block_size = std::min(max_block_size, total_size);
  size_t grid_size = total_size < max_block_size ? 1 : (total_size + max_block_size - 1) / max_block_size;

  auto out = array<value_type>(total_size);
  detail::materialize<<<grid_size, block_size>>>(total_size, std::move(in), out.data());
  return out;
}

template<typename V>
__global__ auto filtered_materialize(size_t total_size, V in, uint8_t* presence, size_t* indices,
                                     typename V::value_type* out) -> void {
  auto id = blockIdx.x * blockDim.x + threadIdx.x;
  if (id < total_size) {
    in.get(id,
           [&](auto v) {
             out[id] = v;
             indices[id] = 1;
             presence[id] = 1;
           },
           [&] {
             indices[id] = 0;
             presence[id] = 0;
           });
  }

  auto tid = threadIdx.x;
  auto stride = 1;
  while (stride != blockDim.x) {
    __syncthreads();

    if (tid % (stride * 2) == 0 && id + stride < total_size) {
      auto i1 = indices[id];
      auto i2 = indices[id + stride];
      indices[id + stride] = i1;
      indices[id] = i1 + i2;
    }
    stride *= 2;
  }
}

__global__ auto filtered_materialize_blocks(size_t total_size, size_t step, size_t* indices) -> void {
  // FIXME: Deduplicate
  auto id = blockIdx.x * blockDim.x + threadIdx.x;
  auto tid = threadIdx.x;
  auto stride = 1;
  while (stride != blockDim.x) {
    __syncthreads();

    if (tid % (stride * 2) == 0 && (id + stride) * step < total_size) {
      auto i1 = indices[id * step];
      auto i2 = indices[(id + stride) * step];
      indices[(id + stride) * step] = i1;
      indices[id * step] = i1 + i2;
    }
    stride *= 2;
  }
}

__global__ auto filtered_remap_blocks(size_t total_size, size_t step, size_t* indices) -> void {
  // FIXME: Deduplicate
  auto id = blockIdx.x * blockDim.x + threadIdx.x;
  indices[0] = 0;
  auto tid = threadIdx.x;
  auto stride = blockDim.x / 2;
  while (stride) {
    __syncthreads();
    if (tid % (stride * 2) == 0 && (id + stride) * step < total_size) {
      indices[(id + stride) * step] += indices[id * step];
    }
    stride /= 2;
  }
}

template<typename T>
__global__ auto filtered_remap(size_t total_size, T* in, uint8_t const* presence, size_t* indices, T* out) -> void {
  auto id = blockIdx.x * blockDim.x + threadIdx.x;
  indices[0] = 0;
  auto tid = threadIdx.x;
  auto stride = blockDim.x / 2;
  while (stride) {
    __syncthreads();
    if (tid % (stride * 2) == 0 && id + stride < total_size) { indices[id + stride] += indices[id]; }
    stride /= 2;
  }

  __syncthreads();
  if (id < total_size && presence[id]) { out[indices[id]] = in[id]; }
}

template<typename V> auto to_gpu_fn_impl(known_size_tag<false>, V in) -> array<typename V::value_type> {
  using value_type = typename V::value_type;
  static constexpr size_t max_block_size = 1024;

  size_t total_size = in.size_bound();
  size_t block_size = std::min(max_block_size, next_pow2(total_size));
  size_t grid_size = total_size < max_block_size ? 1 : (total_size + max_block_size - 1) / max_block_size;

  auto out = array<value_type>(total_size);
  auto indices = array<size_t>(total_size);
  auto presence = array<uint8_t>(total_size);
  detail::filtered_materialize<<<grid_size, block_size>>>(total_size, std::move(in), presence.data(), indices.data(),
                                                          out.data());

  size_t step = 1;
  auto step_grid_size = grid_size;
  while (step_grid_size != 1) {
    auto step_total_size = step_grid_size;

    step *= max_block_size;
    auto step_block_size = std::min(max_block_size, next_pow2(step_total_size));
    step_grid_size = step_total_size < max_block_size ? 1 : (step_total_size + max_block_size - 1) / max_block_size;
    filtered_materialize_blocks<<<step_grid_size, step_block_size>>>(total_size, step, indices.data());
  }

  auto filtered_size = std::array<size_t, 1>{};
  copy(span<size_t>(indices.data(), indices.data() + 1), filtered_size);

  while (step_grid_size != grid_size) {
    auto step_total_size = (total_size + step - 1) / step;

    auto step_block_size = std::min(max_block_size, next_pow2(step_total_size));
    filtered_remap_blocks<<<step_grid_size, step_block_size>>>(total_size, step, indices.data());
    step /= max_block_size;
    step_grid_size = step_total_size;
  }

  auto filtered_out = array<value_type>(filtered_size[0]);
  detail::filtered_remap<<<grid_size, block_size>>>(total_size, out.data(), presence.data(), indices.data(),
                                                    filtered_out.data());
  return filtered_out;
}

} // namespace detail

struct to_gpu {};

template<typename V> auto to_gpu_fn(V&& v) -> array<typename std::decay_t<V>::value_type> {
  auto in = view::detail::to_view(std::forward<V>(v));
  return detail::with_prepared(detail::needs_preparing_tag<in.needs_preparing()>{}, std::move(in), [](auto vin) {
    return detail::to_gpu_fn_impl(detail::known_size_tag<vin.known_size()>{}, std::move(vin));
  });
}

template<typename V> auto operator|(V&& v, to_gpu) {
  return to_gpu_fn(std::forward<V>(v));
}

} // namespace action
} // namespace ranges_gpu
