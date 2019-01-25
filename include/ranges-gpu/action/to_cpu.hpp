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

namespace ranges_gpu {
namespace action {

namespace detail {

template<typename V> __global__ auto materialize(V in, typename V::value_type* out) -> void {
  auto id = blockIdx.x * blockDim.x + threadIdx.x;
  out[id] = in[id];
}

} // namespace detail

struct to_cpu {};

template<typename V> auto to_cpu_fn(V&& v) -> std::vector<typename std::decay_t<V>::value_type> {
  using value_type = typename std::decay_t<V>::value_type;
  auto in = view::detail::to_view(std::forward<V>(v));
  auto out = array<value_type>(in.size());
  detail::materialize<<<in.size(), 1>>>(std::move(in), out.data());
  auto vec = std::vector<value_type>(v.size());
  copy(out, vec);
  return vec;
}

template<typename V> auto operator|(V&& v, to_cpu) {
  return to_cpu_fn(std::forward<V>(v));
}

} // namespace action
} // namespace ranges_gpu
