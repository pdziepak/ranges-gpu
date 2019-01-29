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

#include <array>
#include <memory>
#include <type_traits>
#include <vector>

namespace ranges_gpu {

template<typename T> class span {
  T* begin_ = nullptr;
  T* end_ = nullptr;

public:
  static_assert(std::is_trivially_default_constructible<T>::value, "");

  using value_type = T;

  span() = default;

  __host__ __device__ explicit span(T* first, T* last) noexcept : begin_(first), end_(last) {}

  __device__ T& operator[](size_t idx) const noexcept {
    assert(idx < size());
    return begin_[idx];
  }

  __host__ __device__ T* data() const noexcept { return begin_; }
  __host__ __device__ size_t size() const noexcept { return end_ - begin_; }
};

template<typename SourceValueType, typename Destination> void copy(span<SourceValueType> src, Destination&& dst) {
  static_assert(std::is_same<std::decay_t<SourceValueType>, typename std::decay_t<Destination>::value_type>::value, "");
  assert(src.size() <= dst.size());
  auto ret = cudaMemcpy(dst.data(), src.data(), src.size() * sizeof(SourceValueType), cudaMemcpyDeviceToHost);
  assert(ret == cudaSuccess);
}

template<typename Source, typename DestinationValueType> void copy(Source&& src, span<DestinationValueType> dst) {
  static_assert(std::is_same<DestinationValueType, typename std::decay_t<Source>::value_type>::value, "");
  assert(src.size() <= dst.size());
  auto ret = cudaMemcpy(dst.data(), src.data(), src.size() * sizeof(DestinationValueType), cudaMemcpyHostToDevice);
  assert(ret == cudaSuccess);
}

} // namespace ranges_gpu
