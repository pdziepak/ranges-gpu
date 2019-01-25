#
# Copyright © 2019 Paweł Dziepak
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

cmake_minimum_required(VERSION 3.10)

project(ranges-gpu
  LANGUAGES CXX CUDA
)

include(CTest)

list(APPEND RANGES_GPU_CXX_FLAGS -Wall -Wextra -Werror)
list(APPEND RANGES_GPU_CXX_DEBUG_FLAGS -fsanitize=address -fsanitize=address)
list(APPEND RANGES_GPU_CUDA_DEBUG_FLAGS -G)
list(APPEND RANGES_GPU_CUDA_RELEASE_FLAGS -lineinfo)

if(CMAKE_BUILD_TYPE STREQUAL Debug)
  list(APPEND RANGES_GPU_CXX_FLAGS ${RANGES_GPU_CXX_DEBUG_FLAGS})
  list(APPEND RANGES_GPU_CUDA_FLAGS ${RANGES_GPU_CUDA_DEBUG_FLAGS})
else()
  list(APPEND RANGES_GPU_CXX_FLAGS ${RANGES_GPU_CXX_RELEASE_FLAGS})
  list(APPEND RANGES_GPU_CUDA_FLAGS ${RANGES_GPU_CUDA_RELEASE_FLAGS})
endif()

string(REPLACE ";" "," RANGES_GPU_CUDA_CXX_FLAGS "${RANGES_GPU_CXX_FLAGS}")
list(APPEND RANGES_GPU_CUDA_FLAGS -Xcompiler ${RANGES_GPU_CUDA_CXX_FLAGS})

add_library(ranges-gpu INTERFACE)
target_include_directories(ranges-gpu INTERFACE $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include/>)
target_compile_features(ranges-gpu INTERFACE cxx_std_14)
target_compile_options(ranges-gpu INTERFACE --expt-extended-lambda)

if(BUILD_TESTING)
  add_subdirectory(tests)
endif()
