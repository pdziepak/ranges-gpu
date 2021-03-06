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

find_package(GTest REQUIRED)

function(ranges_gpu_add_test TESTNAME SOURCE)
  add_executable(${TESTNAME} ${SOURCE} ${ARGN})
  target_link_libraries(${TESTNAME} ranges-gpu GTest::GTest GTest::Main ${RANGES_GPU_CXX_FLAGS})
  target_compile_options(${TESTNAME} PRIVATE ${RANGES_GPU_CUDA_FLAGS})
  add_test(${TESTNAME} ${TESTNAME})
endfunction(ranges_gpu_add_test)

ranges_gpu_add_test(array array.cu)
ranges_gpu_add_test(examples examples.cu)
ranges_gpu_add_test(span span.cu)

ranges_gpu_add_test(action_all_of action/all_of.cu)
ranges_gpu_add_test(action_any_of action/any_of.cu)
ranges_gpu_add_test(action_none_of action/none_of.cu)
ranges_gpu_add_test(action_reduce action/reduce.cu)
ranges_gpu_add_test(action_to_cpu action/to_cpu.cu)
ranges_gpu_add_test(action_to_gpu action/to_gpu.cu)

ranges_gpu_add_test(view_iota view/iota.cu)
ranges_gpu_add_test(view_filter view/filter.cu)
ranges_gpu_add_test(view_take view/take.cu)
ranges_gpu_add_test(view_to_gpu view/to_gpu.cu)
ranges_gpu_add_test(view_transform view/transform.cu)
ranges_gpu_add_test(view_zip view/zip.cu)
