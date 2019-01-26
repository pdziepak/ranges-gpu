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

template<typename V, typename T, typename F>
__global__ auto do_reduce(size_t total_size, V in, T init, F fn, T* out) -> void {
  auto id = blockIdx.x * blockDim.x + threadIdx.x;
  if (id < total_size) { out[id] = in[id]; }
  auto tid = threadIdx.x;
  auto stride = blockDim.x / 2;
  while (stride) {
    __syncthreads();
    if (tid < stride && id + stride < total_size) { out[id] = fn(out[id], out[id + stride]); }
    stride /= 2;
  }
  if (id == 0) { out[0] = fn(out[0], init); }
}

template<typename T, typename F>
__global__ auto do_reduce_blocks(size_t total_size, size_t step, F fn, T* out) -> void {
  auto id = blockIdx.x * blockDim.x + threadIdx.x;
  auto tid = threadIdx.x;
  auto stride = blockDim.x / 2;
  while (stride) {
    __syncthreads();
    if (tid < stride && (id + stride) * step < total_size) {
      out[id * step] = fn(out[id * step], out[(id + stride) * step]);
    }
    stride /= 2;
  }
}

template<typename V, typename T, typename F> auto reduce_fn_impl(V v, T t, F fn) -> T {
  static constexpr size_t max_block_size = 1024;

  if (v.size() == 0) { return t; }

  size_t total_size = v.size();
  size_t block_size = std::min(max_block_size, next_pow2(total_size));
  size_t grid_size = total_size < max_block_size ? 1 : (total_size + max_block_size - 1) / max_block_size;

  auto out = array<T>(total_size);
  detail::do_reduce<<<grid_size, block_size>>>(total_size, std::move(v), std::move(t), fn, out.data());

  size_t step = block_size;
  while (grid_size != 1) {
    auto step_total_size = grid_size;

    block_size = std::min(max_block_size, next_pow2(step_total_size));
    grid_size = step_total_size < max_block_size ? 1 : (step_total_size + max_block_size - 1) / max_block_size;

    detail::do_reduce_blocks<<<grid_size, block_size>>>(total_size, step, fn, out.data());
    step *= block_size;
  }

  // FIXME: shouldn't need this array
  auto result = std::array<T, 1>{};
  copy(span<T>(out.data(), out.data() + 1), result);
  return std::move(result[0]);
}

} // namespace detail

template<typename V, typename T, typename F> auto reduce_fn(V&& v, T t, F fn) -> T {
  auto in = view::detail::to_view(std::forward<V>(v));
  return detail::with_prepared(detail::needs_preparing_tag<in.needs_preparing()>{}, std::move(in), [&](auto vin) {
    return detail::reduce_fn_impl(std::move(vin), std::move(t), std::move(fn));
  });
}

namespace detail {

template<typename T, typename F> struct reduce {
  T init_;
  F fn_;
};

template<typename V, typename T, typename F> auto operator|(V&& v, detail::reduce<T, F> r) {
  return reduce_fn(std::forward<V>(v), std::move(r.init_), std::move(r.fn_));
}

} // namespace detail

template<typename T, typename F> auto reduce(T init, F fn) {
  return detail::reduce<T, F>{std::move(init), std::move(fn)};
}

} // namespace action
} // namespace ranges_gpu
