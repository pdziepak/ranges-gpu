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

#include "detail.hpp"

namespace ranges_gpu {

template<typename T> class array {
  struct deleter {
    void operator()(T* ptr) const noexcept {
      auto ret = cudaFree(ptr);
      assert(ret == cudaSuccess);
      (void)ret;
    }
  };

private:
  std::unique_ptr<T[], deleter> data_;
  size_t size_ = 0;

public:
  static_assert(std::is_trivially_default_constructible<T>::value, "");

  using value_type = T;

  array() = default;

  explicit array(size_t n) {
    void* ptr;
    auto ret = cudaMalloc(&ptr, n * sizeof(T));
    if (ret != cudaSuccess) { throw std::bad_alloc(); }
    data_.reset(static_cast<T*>(ptr));
    size_ = n;
  }

  explicit array(T const* ptr, size_t n) : array(n) {
    auto ret = cudaMemcpy(data_.get(), ptr, size() * sizeof(T), cudaMemcpyHostToDevice);
    assert(ret == cudaSuccess);
    (void)ret;
  }

  explicit array(std::vector<T> const& vec) : array(vec.data(), vec.size()) {}
  template<size_t N> explicit array(std::array<T, N> const& arr) : array(arr.data(), N) {}

  array(array&&) noexcept = default;
  array(array const& other) : array(other.size()) {
    auto ret = cudaMemcpy(data_.get(), other.data_.get(), size() * sizeof(T), cudaMemcpyDeviceToDevice);
    (void)ret;
    assert(ret == cudaSuccess);
  }

  T* data() const noexcept { return data_.get(); }
  size_t size() const noexcept { return size_; }

  T* release() && noexcept { return data_.release(); }
};

template<typename SourceValueType, typename Destination>
void copy(array<SourceValueType> const& src, Destination&& dst) {
  static_assert(std::is_same<SourceValueType, typename std::decay_t<Destination>::value_type>::value, "");
  assert(src.size() <= dst.size());
  auto ret = cudaMemcpy(dst.data(), src.data(), src.size() * sizeof(SourceValueType), cudaMemcpyDeviceToHost);
  assert(ret == cudaSuccess);
}

template<typename Source, typename DestinationValueType> void copy(Source&& src, array<DestinationValueType>& dst) {
  static_assert(std::is_same<DestinationValueType, typename std::decay_t<Source>::value_type>::value, "");
  assert(src.size() <= dst.size());
  auto ret = cudaMemcpy(dst.data(), src.data(), src.size() * sizeof(DestinationValueType), cudaMemcpyHostToDevice);
  assert(ret == cudaSuccess);
}

} // namespace ranges_gpu
