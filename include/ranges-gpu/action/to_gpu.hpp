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
#include "ranges-gpu/view/detail.hpp"

#include "detail.hpp"

namespace ranges_gpu {
namespace action {

namespace detail {

template<typename V> __global__ auto materialize(size_t total_size, V in, typename V::value_type* out) -> void {
  auto id = blockIdx.x * blockDim.x + threadIdx.x;
  if (id < total_size) { out[id] = in[id]; }
}

template<typename V> auto to_gpu_fn_impl(V in) -> array<typename V::value_type> {
  using value_type = typename V::value_type;
  static constexpr size_t max_block_size = 1024;

  size_t total_size = in.size();
  size_t block_size = std::min(max_block_size, total_size);
  size_t grid_size = total_size < max_block_size ? 1 : (total_size + max_block_size - 1) / max_block_size;

  auto out = array<value_type>(total_size);
  detail::materialize<<<grid_size, block_size>>>(total_size, std::move(in), out.data());
  return out;
}

} // namespace detail

struct to_gpu {};

template<typename V> auto to_gpu_fn(V&& v) -> array<typename std::decay_t<V>::value_type> {
  auto in = view::detail::to_view(std::forward<V>(v));
  return detail::with_prepared(detail::needs_preparing_tag<in.needs_preparing()>{}, std::move(in),
                               [](auto vin) { return detail::to_gpu_fn_impl(std::move(vin)); });
}

template<typename V> auto operator|(V&& v, to_gpu) {
  return to_gpu_fn(std::forward<V>(v));
}

} // namespace action
} // namespace ranges_gpu
