/*
This file is part of the LC framework for synthesizing high-speed parallel lossless and error-bounded lossy data compression and decompression algorithms for CPUs and GPUs.

BSD 3-Clause License

Copyright (c) 2021-2025, Noushin Azami, Alex Fallin, Brandon Burtchell, Andrew Rodriguez, Benila Jerald, Yiqian Liu, Anju Mongandampulath Akathoott, and Martin Burtscher
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this
   list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its
   contributors may be used to endorse or promote products derived from
   this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

URL: The latest version of this code is available at https://github.com/burtscher/LC-framework.

Sponsor: This code is based upon work supported by the U.S. Department of Energy, Office of Science, Office of Advanced Scientific Research (ASCR), under contract DE-SC0022223.
*/


#ifndef LC_FRAMEWORK_COMMON_H
#define LC_FRAMEWORK_COMMON_H


using byte = unsigned char;

static const int max_stages = 8;  // cannot be more than 8

#include <cstdlib>
#include <cstdio>
#include <string>
#include <cstring>
#include <strings.h>
#include <cassert>
#include <unistd.h>
#include <limits>
#include <algorithm>
#include <vector>
#include <map>
#include <cmath>
#include <ctime>
#include <regex>
#include <stdexcept>
#include <sys/time.h>


#if defined(_OPENMP)
#include <omp.h>
#endif

#include "include/consts.h"
#ifndef USE_GPU
  #ifndef USE_CPU
  //no CPU and no GPU
  #else
  #include "preprocessors/include/CPUpreprocessors.h"
  #include "components/include/CPUcomponents.h"
  #endif
#else
  #include <cuda.h>
  #if !defined(__HIPCC__)
  #include <cuda/std/limits>
  #endif
  #if defined(__CUDA_ARCH__)
  #include <cuda/atomic>  // a CUDA-only library that cannot be automatically replaced by HIPIFY
  #endif
  #include "include/macros.h"
  #include "include/max_reduction.h"
  #include "include/max_scan.h"
  #include "include/prefix_sum.h"
  #include "include/sum_reduction.h"

  #ifndef USE_CPU
  #include "preprocessors/include/GPUpreprocessors.h"
  #include "components/include/GPUcomponents.h"
  #else
  #include "preprocessors/include/preprocessors.h"
  #include "components/include/components.h"
  #endif
#endif
#include "verifiers/include/verifiers.h"


static void verify(const long long size, const byte* const recon, const byte* const orig, std::vector<std::pair<byte, std::vector<double>>> verifs)
{
  for (int i = 0; i < verifs.size(); i++) {
    std::vector<double> params = verifs[i].second;
    switch (verifs[i].first) {
      default: fprintf(stderr, "ERROR: unknown verifier\n\n"); throw std::runtime_error("LC error"); break;
      /*##switch-verify-beg##*/
      case v_MAXNOA_f32: MAXNOA_f32(size, recon, orig, params.size(), params.data()); break;
      case v_PSNR_f32: PSNR_f32(size, recon, orig, params.size(), params.data()); break;
      case v_MAXABS_f64: MAXABS_f64(size, recon, orig, params.size(), params.data()); break;
      case v_MSE_f64: MSE_f64(size, recon, orig, params.size(), params.data()); break;
      case v_PSNR_f64: PSNR_f64(size, recon, orig, params.size(), params.data()); break;
      case v_MAXNOA_f64: MAXNOA_f64(size, recon, orig, params.size(), params.data()); break;
      case v_MAXREL_f32: MAXREL_f32(size, recon, orig, params.size(), params.data()); break;
      case v_MAXABS_f32: MAXABS_f32(size, recon, orig, params.size(), params.data()); break;
      case v_PASS: PASS(size, recon, orig, params.size(), params.data()); break;
      case v_MSE_f32: MSE_f32(size, recon, orig, params.size(), params.data()); break;
      case v_LOSSLESS: LOSSLESS(size, recon, orig, params.size(), params.data()); break;
      case v_MAXREL_f64: MAXREL_f64(size, recon, orig, params.size(), params.data()); break;
      /*##switch-verify-end##*/
    }
  }
}


#ifdef USE_GPU
static void d_preprocess_encode(long long& dpreencsize, byte*& dpreencdata, std::vector<std::pair<byte, std::vector<double>>> prepros)
{
  for (int i = 0; i < prepros.size(); i++) {
    std::vector<double> params = prepros[i].second;
    switch (prepros[i].first) {
      default: fprintf(stderr, "ERROR: unknown preprocessor\n\n"); throw std::runtime_error("LC error"); break;
      /*##switch-device-preprocess-encode-beg##*/

      // code will be automatically inserted

      /*##switch-device-preprocess-encode-end##*/
    }
  }
}


static void d_preprocess_decode(long long& dpredecsize, byte*& dpredecdata, std::vector<std::pair<byte, std::vector<double>>> prepros)
{
  for (int i = prepros.size() - 1; i >= 0; i--) {
    std::vector<double> params = prepros[i].second;
    switch (prepros[i].first) {
      default: fprintf(stderr, "ERROR: unknown preprocessor\n\n"); throw std::runtime_error("LC error"); break;
      /*##switch-device-preprocess-decode-beg##*/

      // code will be automatically inserted

      /*##switch-device-preprocess-decode-end##*/
    }
  }
}


#if defined(__CUDA_ARCH__)

template <typename T>
__device__ inline T atomicRead(T* const addr)
{
  return ((cuda::atomic<T>*)addr)->load(cuda::memory_order_relaxed);
}


template <typename T>
__device__ inline void atomicWrite(T* const addr, const T val)
{
  ((cuda::atomic<T>*)addr)->store(val, cuda::memory_order_relaxed);
}

#else

template <typename T>
__device__ inline T atomicRead(T* const addr)
{
  return *((volatile T*)addr);  // AMD hack
}


template <typename T>
__device__ inline void atomicWrite(T* const addr, const T val)
{
  *((volatile T*)addr) = val;  // AMD hack
}

#endif /* defined(__CUDA_ARCH__) */
#endif /* USE_GPU */


#ifdef USE_CPU
static void h_preprocess_encode(long long& hpreencsize, byte*& hpreencdata, std::vector<std::pair<byte, std::vector<double>>> prepros)
{
  for (int i = 0; i < prepros.size(); i++) {
    std::vector<double> params = prepros[i].second;
    switch (prepros[i].first) {
      default: fprintf(stderr, "ERROR: unknown preprocessor\n\n"); throw std::runtime_error("LC error"); break;
      /*##switch-host-preprocess-encode-beg##*/
      case QUANT_NOA_0_f64: h_QUANT_NOA_0_f64(hpreencsize, hpreencdata, params.size(), params.data()); break;
      case LOR1D_i32: h_LOR1D_i32(hpreencsize, hpreencdata, params.size(), params.data()); break;
      case QUANT_NOA_R_f32: h_QUANT_NOA_R_f32(hpreencsize, hpreencdata, params.size(), params.data()); break;
      case QUANT_NOA_0_f32: h_QUANT_NOA_0_f32(hpreencsize, hpreencdata, params.size(), params.data()); break;
      case QUANT_ABS_0_f32: h_QUANT_ABS_0_f32(hpreencsize, hpreencdata, params.size(), params.data()); break;
      case QUANT_REL_0_f32: h_QUANT_REL_0_f32(hpreencsize, hpreencdata, params.size(), params.data()); break;
      case QUANT_REL_R_f64: h_QUANT_REL_R_f64(hpreencsize, hpreencdata, params.size(), params.data()); break;
      case QUANT_ABS_0_f64: h_QUANT_ABS_0_f64(hpreencsize, hpreencdata, params.size(), params.data()); break;
      case QUANT_REL_R_f32: h_QUANT_REL_R_f32(hpreencsize, hpreencdata, params.size(), params.data()); break;
      case QUANT_ABS_R_f32: h_QUANT_ABS_R_f32(hpreencsize, hpreencdata, params.size(), params.data()); break;
      case QUANT_NOA_R_f64: h_QUANT_NOA_R_f64(hpreencsize, hpreencdata, params.size(), params.data()); break;
      case QUANT_ABS_R_f64: h_QUANT_ABS_R_f64(hpreencsize, hpreencdata, params.size(), params.data()); break;
      case QUANT_REL_0_f64: h_QUANT_REL_0_f64(hpreencsize, hpreencdata, params.size(), params.data()); break;
      /*##switch-host-preprocess-encode-end##*/
    }
  }
}


static void h_preprocess_decode(long long& hpredecsize, byte*& hpredecdata, std::vector<std::pair<byte, std::vector<double>>> prepros)
{
  for (int i = prepros.size() - 1; i >= 0; i--) {
    std::vector<double> params = prepros[i].second;
    switch (prepros[i].first) {
      default: fprintf(stderr, "ERROR: unknown preprocessor\n\n"); throw std::runtime_error("LC error"); break;
      /*##switch-host-preprocess-decode-beg##*/
      case QUANT_NOA_0_f64: h_iQUANT_NOA_0_f64(hpredecsize, hpredecdata, params.size(), params.data()); break;
      case LOR1D_i32: h_iLOR1D_i32(hpredecsize, hpredecdata, params.size(), params.data()); break;
      case QUANT_NOA_R_f32: h_iQUANT_NOA_R_f32(hpredecsize, hpredecdata, params.size(), params.data()); break;
      case QUANT_NOA_0_f32: h_iQUANT_NOA_0_f32(hpredecsize, hpredecdata, params.size(), params.data()); break;
      case QUANT_ABS_0_f32: h_iQUANT_ABS_0_f32(hpredecsize, hpredecdata, params.size(), params.data()); break;
      case QUANT_REL_0_f32: h_iQUANT_REL_0_f32(hpredecsize, hpredecdata, params.size(), params.data()); break;
      case QUANT_REL_R_f64: h_iQUANT_REL_R_f64(hpredecsize, hpredecdata, params.size(), params.data()); break;
      case QUANT_ABS_0_f64: h_iQUANT_ABS_0_f64(hpredecsize, hpredecdata, params.size(), params.data()); break;
      case QUANT_REL_R_f32: h_iQUANT_REL_R_f32(hpredecsize, hpredecdata, params.size(), params.data()); break;
      case QUANT_ABS_R_f32: h_iQUANT_ABS_R_f32(hpredecsize, hpredecdata, params.size(), params.data()); break;
      case QUANT_NOA_R_f64: h_iQUANT_NOA_R_f64(hpredecsize, hpredecdata, params.size(), params.data()); break;
      case QUANT_ABS_R_f64: h_iQUANT_ABS_R_f64(hpredecsize, hpredecdata, params.size(), params.data()); break;
      case QUANT_REL_0_f64: h_iQUANT_REL_0_f64(hpredecsize, hpredecdata, params.size(), params.data()); break;
      /*##switch-host-preprocess-decode-end##*/
    }
  }
}
#endif


#ifdef USE_GPU
static void __global__ initBestSize(unsigned short* const bestSize, const int chunks)
{
  if ((threadIdx.x == 0) && (WS != warpSize)) {printf("ERROR: WS must be %d\n\n", warpSize); __trap();}  // debugging only
  for (int i = threadIdx.x; i < chunks; i += TPB) {
    bestSize[i] = CS;
  }
}


static void __global__ dbestChunkSize(const byte* const __restrict__ input, unsigned short* const __restrict__ bestSize)
{
  int* const head_in = (int*)input;
  const int outsize = head_in[0];
  const int chunks = (outsize + CS - 1) / CS;  // round up
  unsigned short* const size_in = (unsigned short*)&head_in[1];
  for (int chunkID = threadIdx.x; chunkID < chunks; chunkID += TPB) {
    bestSize[chunkID] = min(bestSize[chunkID], size_in[chunkID]);
  }
}


static void __global__ dcompareData(const long long size, const byte* const __restrict__ data1, const byte* const __restrict__ data2, unsigned long long* const __restrict__ min_loc)
{
  const long long i = threadIdx.x + (long long)blockIdx.x * TPB;
  if (i < size) {
    if (data1[i] != data2[i]) atomicMin(min_loc, i);
  }
}


static __device__ unsigned long long g_chunk_counter;


static __global__ void d_reset()
{
  g_chunk_counter = 0LL;
}

static inline __device__ void propagate_carry(const int value, const long long chunkID, long long* const __restrict__ fullcarry, long long* const __restrict__ s_fullc)
{
  if (threadIdx.x == TPB - 1) {  // last thread
    atomicWrite(&fullcarry[chunkID], (chunkID == 0) ? (long long)value : (long long)-value);
  }

  if (chunkID != 0) {
    if (threadIdx.x + WS >= TPB) {  // last warp
      const int lane = threadIdx.x % WS;
      const long long cidm1ml = chunkID - 1 - lane;
      long long val = -1;
      __syncwarp();  // not optional
      do {
        if (cidm1ml >= 0) {
          val = atomicRead(&fullcarry[cidm1ml]);
        }
      } while ((__any(val == 0)) || (__all(val <= 0)));
#if defined(WS) && (WS == 64)
      const long long mask = __ballot(val > 0);
      const int pos = __ffsll(mask) - 1;
#else
      const int mask = __ballot(val > 0);
      const int pos = __ffs(mask) - 1;
#endif
      long long partc = (lane < pos) ? -val : 0;
      partc += __shfl_xor(partc, 1);
      partc += __shfl_xor(partc, 2);
      partc += __shfl_xor(partc, 4);
      partc += __shfl_xor(partc, 8);
      partc += __shfl_xor(partc, 16);
#if defined(WS) && (WS == 64)
      partc += __shfl_xor(partc, 32);
#endif
      if (lane == pos) {
        const long long fullc = partc + val;
        atomicWrite(&fullcarry[chunkID], fullc + value);
        *s_fullc = fullc;
      }
    }
  }
}


// copy (len) bytes from shared memory (source) to global memory (destination)
// source must we word aligned
static inline __device__ void s2g(void* const __restrict__ destination, const void* const __restrict__ source, const int len)
{
  const int tid = threadIdx.x;
  const byte* const __restrict__ input = (byte*)source;
  byte* const __restrict__ output = (byte*)destination;
  if (len < 128) {
    if (tid < len) output[tid] = input[tid];
  } else {
    const int nonaligned = (int)(size_t)output;
    const int wordaligned = (nonaligned + 3) & ~3;
    const int linealigned = (nonaligned + 127) & ~127;
    const int bcnt = wordaligned - nonaligned;
    const int wcnt = (linealigned - wordaligned) / 4;
    const int* const __restrict__ in_w = (int*)input;
    if (bcnt == 0) {
      int* const __restrict__ out_w = (int*)output;
      if (tid < wcnt) out_w[tid] = in_w[tid];
      for (int i = tid + wcnt; i < len / 4; i += TPB) {
        out_w[i] = in_w[i];
      }
      if (tid < (len & 3)) {
        const int i = len - 1 - tid;
        output[i] = input[i];
      }
    } else {
      const int shift = bcnt * 8;
      const int rlen = len - bcnt;
      int* const __restrict__ out_w = (int*)&output[bcnt];
      if (tid < bcnt) output[tid] = input[tid];
      if (tid < wcnt) out_w[tid] = __funnelshift_r(in_w[tid], in_w[tid + 1], shift);
      for (int i = tid + wcnt; i < rlen / 4; i += TPB) {
        out_w[i] = __funnelshift_r(in_w[i], in_w[i + 1], shift);
      }
      if (tid < (rlen & 3)) {
        const int i = len - 1 - tid;
        output[i] = input[i];
      }
    }
  }
}


// copy (len) bytes from global memory (source) to shared memory (destination) using separate shared memory buffer (temp)
// destination and temp must we word aligned, accesses up to CS + 3 bytes in temp
static inline __device__ void g2s(void* const __restrict__ destination, const void* const __restrict__ source, const int len, void* const __restrict__ temp)
{
  const int tid = threadIdx.x;
  const byte* const __restrict__ input = (byte*)source;
  if (len < 128) {
    byte* const __restrict__ output = (byte*)destination;
    if (tid < len) output[tid] = input[tid];
  } else {
    const int nonaligned = (int)(size_t)input;
    const int wordaligned = (nonaligned + 3) & ~3;
    const int linealigned = (nonaligned + 127) & ~127;
    const int bcnt = wordaligned - nonaligned;
    const int wcnt = (linealigned - wordaligned) / 4;
    int* const __restrict__ out_w = (int*)destination;
    if (bcnt == 0) {
      const int* const __restrict__ in_w = (int*)input;
      byte* const __restrict__ out = (byte*)destination;
      if (tid < wcnt) out_w[tid] = in_w[tid];
      for (int i = tid + wcnt; i < len / 4; i += TPB) {
        out_w[i] = in_w[i];
      }
      if (tid < (len & 3)) {
        const int i = len - 1 - tid;
        out[i] = input[i];
      }
    } else {
      const int offs = 4 - bcnt;  //(4 - bcnt) & 3;
      const int shift = offs * 8;
      const int rlen = len - bcnt;
      const int* const __restrict__ in_w = (int*)&input[bcnt];
      byte* const __restrict__ buffer = (byte*)temp;
      byte* const __restrict__ buf = (byte*)&buffer[offs];
      int* __restrict__ buf_w = (int*)&buffer[4];  //(int*)&buffer[(bcnt + 3) & 4];
      if (tid < bcnt) buf[tid] = input[tid];
      if (tid < wcnt) buf_w[tid] = in_w[tid];
      for (int i = tid + wcnt; i < rlen / 4; i += TPB) {
        buf_w[i] = in_w[i];
      }
      if (tid < (rlen & 3)) {
        const int i = len - 1 - tid;
        buf[i] = input[i];
      }
      __syncthreads();
      buf_w = (int*)buffer;
      for (int i = tid; i < (len + 3) / 4; i += TPB) {
        out_w[i] = __funnelshift_r(buf_w[i], buf_w[i + 1], shift);
      }
    }
  }
}


#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ == 800)
static __global__ __launch_bounds__(TPB, 3)
#else
static __global__ __launch_bounds__(TPB, 2)
#endif
void d_encode(const unsigned long long chain, const byte* const __restrict__ input, const long long insize, byte* const __restrict__ output, long long* const __restrict__ outsize, long long* const __restrict__ fullcarry)
{
  // allocate shared memory buffer
  __shared__ long long chunk [3 * (CS / sizeof(long long))];

  // split into 3 shared memory buffers
  byte* in = (byte*)&chunk[0 * (CS / sizeof(long long))];
  byte* out = (byte*)&chunk[1 * (CS / sizeof(long long))];
  byte* const temp = (byte*)&chunk[2 * (CS / sizeof(long long))];

  // initialize
  const int tid = threadIdx.x;
  const long long last = 3 * (CS / sizeof(long long)) - 2 - WS;
  const long long chunks = (insize + CS - 1) / CS;  // round up
  long long* const head_out = (long long*)output;
  unsigned short* const size_out = (unsigned short*)&head_out[1];
  byte* const data_out = (byte*)&size_out[chunks];

  // loop over chunks
  do {
    // assign work dynamically
    if (tid == 0) chunk[last] = atomicAdd(&g_chunk_counter, 1LL);
    __syncthreads();  // chunk[last] produced, chunk consumed

    // terminate if done
    const long long chunkID = chunk[last];
    const long long base = chunkID * CS;
    if (base >= insize) break;

    // load chunk
    const int osize = (int)min((long long)CS, insize - base);
    long long* const input_l = (long long*)&input[base];
    long long* const out_l = (long long*)out;
    for (int i = tid; i < osize / 8; i += TPB) {
      out_l[i] = input_l[i];
    }
    const int extra = osize % 8;
    if (tid < extra) out[(long long)osize - (long long)extra + (long long)tid] = input[base + (long long)osize - (long long)extra + (long long)tid];

    // encode chunk
    int csize = osize;
    bool good = true;
    unsigned long long pipeline = chain;
    while ((pipeline != 0) && good) {
      __syncthreads();  // "out" produced, chunk[last] consumed
      byte* tmp = in; in = out; out = tmp;
      switch (pipeline & 0xff) {
        default: {byte* tmp = in; in = out; out = tmp;} break;
        /*##switch-device-encode-beg##*/

        // code will be automatically inserted

        /*##switch-device-encode-end##*/
      }
      pipeline >>= 8;
    }
    __syncthreads();  // "temp" and "out" done

    // handle carry
    if (!good || (csize >= osize)) csize = osize;
    propagate_carry(csize, chunkID, fullcarry, (long long*)temp);

    // reload chunk if incompressible
    if (tid == 0) size_out[chunkID] = csize;
    if (csize == osize) {
      // store original data
      long long* const out_l = (long long*)out;
      for (long long i = tid; i < osize / 8; i += TPB) {
        out_l[i] = input_l[i];
      }
      const int extra = osize % 8;
      if (tid < extra) out[(long long)osize - (long long)extra + (long long)tid] = input[base + (long long)osize - (long long)extra + (long long)tid];
    }
    __syncthreads();  // "out" done, temp produced

    // store chunk
    const long long offs = (chunkID == 0) ? 0 : *((long long*)temp);
    s2g(&data_out[offs], out, csize);

    // finalize if last chunk
    if ((tid == 0) && (base + CS >= insize)) {
      // output header
      head_out[0] = insize;
      // compute compressed size
      *outsize = &data_out[fullcarry[chunkID]] - output;
    }
  } while (true);
}


#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ == 800)
static __global__ __launch_bounds__(TPB, 3)
#else
static __global__ __launch_bounds__(TPB, 2)
#endif
void d_decode(const unsigned long long chain, const byte* const __restrict__ input, byte* const __restrict__ output, long long* const __restrict__ g_outsize)
{
  // allocate shared memory buffer
  __shared__ long long chunk [3 * (CS / sizeof(long long))];
  const int last = 3 * (CS / sizeof(long long)) - 2 - WS;

  // input header
  long long* const head_in = (long long*)input;
  const long long outsize = head_in[0];

  // initialize
  const long long chunks = (outsize + CS - 1) / CS;  // round up
  unsigned short* const size_in = (unsigned short*)&head_in[1];
  byte* const data_in = (byte*)&size_in[chunks];

  // loop over chunks
  const int tid = threadIdx.x;
  long long prevChunkID = 0;
  long long prevOffset = 0;
  do {
    // assign work dynamically
    if (tid == 0) chunk[last] = atomicAdd(&g_chunk_counter, 1LL);
    __syncthreads();  // chunk[last] produced, chunk consumed

    // terminate if done
    const long long chunkID = chunk[last];
    const long long base = chunkID * CS;
    if (base >= outsize) break;

    // compute sum of all prior csizes (start where left off in previous iteration)
    long long sum = 0;
    for (long long i = prevChunkID + tid; i < chunkID; i += TPB) {
      sum += (long long)size_in[i];
    }
    int csize = (int)size_in[chunkID];
    const long long offs = prevOffset + block_sum_reduction(sum, (long long*)&chunk[last + 1]);
    prevChunkID = chunkID;
    prevOffset = offs;

    // create the 3 shared memory buffers
    byte* in = (byte*)&chunk[0 * (CS / sizeof(long long))];
    byte* out = (byte*)&chunk[1 * (CS / sizeof(long long))];
    byte* temp = (byte*)&chunk[2 * (CS / sizeof(long long))];

    // load chunk
    g2s(in, &data_in[offs], csize, out);
    byte* tmp = in; in = out; out = tmp;
    __syncthreads();  // chunk produced, chunk[last] consumed

    // decode
    const int osize = (int)min((long long)CS, outsize - base);
    if (csize < osize) {
      unsigned long long pipeline = chain;
      while (pipeline != 0) {
        byte* tmp = in; in = out; out = tmp;
        switch (pipeline >> 56) {
          default: {byte* tmp = in; in = out; out = tmp;} break;
          /*##switch-device-decode-beg##*/

          // code will be automatically inserted

          /*##switch-device-decode-end##*/
        }
        __syncthreads();  // chunk transformed
        pipeline <<= 8;
      }
    }

    if (csize != osize) {printf("ERROR: csize %d doesn't match osize %d in chunk %lld\n\n", csize, osize, chunkID); __trap();}
    long long* const output_l = (long long*)&output[base];
    long long* const out_l = (long long*)out;
    for (int i = tid; i < osize / 8; i += TPB) {
      output_l[i] = out_l[i];
    }
    const int extra = osize % 8;
    if (tid < extra) output[base + osize - extra + tid] = out[osize - extra + tid];
  } while (true);

  if ((blockIdx.x == 0) && (tid == 0)) {
    *g_outsize = outsize;
  }
}

#endif


#ifdef USE_CPU
static void h_encode(const unsigned long long chain, const byte* const __restrict__ input, const long long insize, byte* const __restrict__ output, long long& outsize)
{
  // 检查最后一个组件是否为HANS
  bool has_hans_at_end = false;
  int hans_type = 0;
  unsigned long long temp_chain = chain;
  unsigned long long last_component = 0;
  
  while (temp_chain != 0) {
    last_component = temp_chain & 0xff;
    temp_chain >>= 8;
  }
  
  if (last_component == HANS_1 || last_component == HANS_2) {
    has_hans_at_end = true;
    hans_type = last_component;
  }
  
  // 如果最后一个组件是HANS，则移除它进行常规处理
  unsigned long long processing_chain = chain;
  if (has_hans_at_end) {
    processing_chain = chain & ~(last_component << ((sizeof(unsigned long long) * 8) - 8));
  }
  
  // initialize
  const long long chunks = (insize + CS - 1) / CS;  // round up
  long long* const head_out = (long long*)output;
  unsigned short* const size_out = (unsigned short*)&head_out[1];
  byte* const data_out = (byte*)&size_out[chunks];
  long long* const carry = new long long [chunks];
  memset(carry, 0, chunks * sizeof(long long));

  // process chunks in parallel
  #pragma omp parallel for schedule(dynamic, 1)
  for (long long chunkID = 0; chunkID < chunks; chunkID++) {
    // load chunk
    long long chunk1 [CS / sizeof(long long)];
    long long chunk2 [CS / sizeof(long long)];
    byte* in = (byte*)chunk1;
    byte* out = (byte*)chunk2;
    const long long base = chunkID * CS;
    const int osize = (int)std::min((long long)CS, insize - base);
    memcpy(out, &input[base], osize);

    // encode chunk
    int csize = osize;
    bool good = true;
    unsigned long long pipeline = processing_chain;  // 使用不包含HANS的处理链
    while ((pipeline != 0) && good) {
      std::swap(in, out);
      switch (pipeline & 0xff) {
        default: std::swap(in, out); break;
        /*##switch-host-encode-beg##*/
        case BIT_1: good = h_BIT_1(csize, in, out); break;
        case BIT_2: good = h_BIT_2(csize, in, out); break;
        case BIT_4: good = h_BIT_4(csize, in, out); break;
        case BIT_8: good = h_BIT_8(csize, in, out); break;
        case CLOG_1: good = h_CLOG_1(csize, in, out); break;
        case CLOG_2: good = h_CLOG_2(csize, in, out); break;
        case CLOG_4: good = h_CLOG_4(csize, in, out); break;
        case CLOG_8: good = h_CLOG_8(csize, in, out); break;
        case DBEFS_4: good = h_DBEFS_4(csize, in, out); break;
        case DBEFS_8: good = h_DBEFS_8(csize, in, out); break;
        case DBESF_4: good = h_DBESF_4(csize, in, out); break;
        case DBESF_8: good = h_DBESF_8(csize, in, out); break;
        case DIFFMS_1: good = h_DIFFMS_1(csize, in, out); break;
        case DIFFMS_2: good = h_DIFFMS_2(csize, in, out); break;
        case DIFFMS_4: good = h_DIFFMS_4(csize, in, out); break;
        case DIFFMS_8: good = h_DIFFMS_8(csize, in, out); break;
        case DIFFNB_1: good = h_DIFFNB_1(csize, in, out); break;
        case DIFFNB_2: good = h_DIFFNB_2(csize, in, out); break;
        case DIFFNB_4: good = h_DIFFNB_4(csize, in, out); break;
        case DIFFNB_8: good = h_DIFFNB_8(csize, in, out); break;
        case DIFF_1: good = h_DIFF_1(csize, in, out); break;
        case DIFF_2: good = h_DIFF_2(csize, in, out); break;
        case DIFF_4: good = h_DIFF_4(csize, in, out); break;
        case DIFF_8: good = h_DIFF_8(csize, in, out); break;
        // 移除HANS处理，因为我们会在最后单独处理
        // case HANS_1: good = h_HANS_1(csize, in, out); break;
        // case HANS_2: good = h_HANS_2(csize, in, out); break;
        case HCLOG_1: good = h_HCLOG_1(csize, in, out); break;
        case HCLOG_2: good = h_HCLOG_2(csize, in, out); break;
        case HCLOG_4: good = h_HCLOG_4(csize, in, out); break;
        case HCLOG_8: good = h_HCLOG_8(csize, in, out); break;
        case RARE_1: good = h_RARE_1(csize, in, out); break;
        case RARE_2: good = h_RARE_2(csize, in, out); break;
        case RARE_4: good = h_RARE_4(csize, in, out); break;
        case RARE_8: good = h_RARE_8(csize, in, out); break;
        case RAZE_1: good = h_RAZE_1(csize, in, out); break;
        case RAZE_2: good = h_RAZE_2(csize, in, out); break;
        case RAZE_4: good = h_RAZE_4(csize, in, out); break;
        case RAZE_8: good = h_RAZE_8(csize, in, out); break;
        case RLE_1: good = h_RLE_1(csize, in, out); break;
        case RLE_2: good = h_RLE_2(csize, in, out); break;
        case RLE_4: good = h_RLE_4(csize, in, out); break;
        case RLE_8: good = h_RLE_8(csize, in, out); break;
        case RRE_1: good = h_RRE_1(csize, in, out); break;
        case RRE_2: good = h_RRE_2(csize, in, out); break;
        case RRE_4: good = h_RRE_4(csize, in, out); break;
        case RRE_8: good = h_RRE_8(csize, in, out); break;
        case RZE_1: good = h_RZE_1(csize, in, out); break;
        case RZE_2: good = h_RZE_2(csize, in, out); break;
        case RZE_4: good = h_RZE_4(csize, in, out); break;
        case RZE_8: good = h_RZE_8(csize, in, out); break;
        case TCMS_1: good = h_TCMS_1(csize, in, out); break;
        case TCMS_2: good = h_TCMS_2(csize, in, out); break;
        case TCMS_4: good = h_TCMS_4(csize, in, out); break;
        case TCMS_8: good = h_TCMS_8(csize, in, out); break;
        case TCNB_1: good = h_TCNB_1(csize, in, out); break;
        case TCNB_2: good = h_TCNB_2(csize, in, out); break;
        case TCNB_4: good = h_TCNB_4(csize, in, out); break;
        case TCNB_8: good = h_TCNB_8(csize, in, out); break;
        case TUPL12_1: good = h_TUPL12_1(csize, in, out); break;
        case TUPL2_1: good = h_TUPL2_1(csize, in, out); break;
        case TUPL2_2: good = h_TUPL2_2(csize, in, out); break;
        case TUPL2_4: good = h_TUPL2_4(csize, in, out); break;
        case TUPL3_1: good = h_TUPL3_1(csize, in, out); break;
        case TUPL3_2: good = h_TUPL3_2(csize, in, out); break;
        case TUPL3_8: good = h_TUPL3_8(csize, in, out); break;
        case TUPL4_1: good = h_TUPL4_1(csize, in, out); break;
        case TUPL4_2: good = h_TUPL4_2(csize, in, out); break;
        case TUPL6_1: good = h_TUPL6_1(csize, in, out); break;
        case TUPL6_2: good = h_TUPL6_2(csize, in, out); break;
        case TUPL6_4: good = h_TUPL6_4(csize, in, out); break;
        case TUPL6_8: good = h_TUPL6_8(csize, in, out); break;
        case TUPL8_1: good = h_TUPL8_1(csize, in, out); break;
        /*##switch-host-encode-end##*/
      }
      pipeline >>= 8;
    }

    // handle carry and store chunk
    long long offs = 0LL;
    if (chunkID > 0) {
      do {
        #pragma omp atomic read
        offs = carry[chunkID - 1];
      } while (offs == 0);
      #pragma omp flush
    }

    // store compressed data
    #pragma omp atomic write
    carry[chunkID] = (offs + (long long)csize);
    size_out[chunkID] = csize;
    memcpy(&data_out[offs], out, csize);

  }

  // 如果最后一个组件是HANS，则进行整体处理
  if (has_hans_at_end) {
    // 计算当前压缩后的总大小
    long long compressed_size = carry[chunks - 1];
    
    // 创建临时缓冲区
    byte* whole_buffer1 = new byte[compressed_size];
    byte* whole_buffer2 = new byte[compressed_size];
    
    // 从data_out复制所有压缩数据到临时缓冲区
    memcpy(whole_buffer1, data_out, compressed_size);
    
    // 处理HANS
    uint hans_size = (uint)compressed_size;
    bool hans_good = false;
    
    if (hans_type == HANS_1) {
      hans_good = h_HANS_1(hans_size, whole_buffer1, whole_buffer2);
    } else if (hans_type == HANS_2) {
      hans_good = h_HANS_2(hans_size, whole_buffer1, whole_buffer2);
    }

      // 更新头部信息
      head_out[0] = insize;  // 保持原始大小不变
      
      // 重新计算每个块的大小和偏移
      long long new_chunks = (hans_size + CS - 1) / CS;  // 新的块数
      
      // 清空原有的carry和size_out
      memset(carry, 0, chunks * sizeof(long long));
      
      // 将HANS处理后的数据写回data_out
      memcpy(data_out, whole_buffer2, hans_size);
      
      // 更新每个块的大小和偏移
      long long offset = 0;
      for (long long i = 0; i < chunks; i++) {
        if (i < new_chunks) {
          const int chunk_size = (int)std::min((long long)CS, hans_size - i * CS);
          if (chunk_size <= 0) {
            size_out[i] = 0;
            carry[i] = offset;
          } else {
            size_out[i] = chunk_size;
            carry[i] = offset + chunk_size;
            offset = carry[i];
          }
        } else {
          size_out[i] = 0;
          carry[i] = offset;
        }
      }
      
      // 更新总大小
      outsize = hans_size + ((byte*)size_out - output);

    
    // 清理临时缓冲区
    delete[] whole_buffer1;
    delete[] whole_buffer2;
  } else {
    // 没有HANS，直接输出
    outsize = &data_out[carry[chunks - 1]] - output;
  }

  // output header
  head_out[0] = insize;

  // finish
  delete[] carry;
}


static void h_encode(const unsigned long long chain, const byte* const __restrict__ input, const long long insize, byte* const __restrict__ output, long long& outsize, const int n_threads)
{
  #ifdef _OPENMP
  const int before = omp_get_max_threads();
  omp_set_num_threads(n_threads);
  #endif

  h_encode(chain, input, insize, output, outsize);

  #ifdef _OPENMP
  omp_set_num_threads(before);
  #endif
}


static void hbestChunkSize(const byte* const __restrict__ input, unsigned short* const __restrict__ bestSize)
{
  long long* const head_in = (long long*)input;
  const long long outsize = head_in[0];
  const long long chunks = (outsize + CS - 1) / CS;  // round up
  unsigned short* const size_in = (unsigned short*)&head_in[1];
  for (int chunkID = 0; chunkID < chunks; chunkID++) {
    bestSize[chunkID] = std::min(bestSize[chunkID], size_in[chunkID]);
  }
}


static void h_decode(const unsigned long long chain, const byte* const __restrict__ input, byte* const __restrict__ output, long long& outsize)
{
  // input header
  long long* const head_in = (long long*)input;
  outsize = head_in[0];

  // initialize
  const long long chunks = (outsize + CS - 1) / CS;  // round up
  unsigned short* const size_in = (unsigned short*)&head_in[1];
  byte* const data_in = (byte*)&size_in[chunks];
  long long* const start = new long long [chunks];

  // convert chunk sizes into starting positions
  long long pfs = 0;
  for (long long chunkID = 0; chunkID < chunks; chunkID++) {
    start[chunkID] = pfs;
    pfs += (long long)size_in[chunkID];
  }

  // process chunks in parallel
  #pragma omp parallel for schedule(dynamic, 1)
  for (long long chunkID = 0; chunkID < chunks; chunkID++) {
    // load chunk
    long long chunk1 [CS / sizeof(long long)];
    long long chunk2 [CS / sizeof(long long)];
    byte* in = (byte*)chunk1;
    byte* out = (byte*)chunk2;
    const long long base = chunkID * CS;
    const int osize = (int)std::min((long long)CS, outsize - base);
    int csize = size_in[chunkID];
    if (csize == osize) {
      // simply copy
      memcpy(&output[base], &data_in[start[chunkID]], osize);
    } else {
      // decompress
      memcpy(out, &data_in[start[chunkID]], csize);

      // decode
      unsigned long long pipeline = chain;
      while (pipeline != 0) {
        std::swap(in, out);
        switch (pipeline >> 56) {
          default: std::swap(in, out); break;
          /*##switch-host-decode-beg##*/
          case BIT_1: h_iBIT_1(csize, in, out); break;
          case BIT_2: h_iBIT_2(csize, in, out); break;
          case BIT_4: h_iBIT_4(csize, in, out); break;
          case BIT_8: h_iBIT_8(csize, in, out); break;
          case CLOG_1: h_iCLOG_1(csize, in, out); break;
          case CLOG_2: h_iCLOG_2(csize, in, out); break;
          case CLOG_4: h_iCLOG_4(csize, in, out); break;
          case CLOG_8: h_iCLOG_8(csize, in, out); break;
          case DBEFS_4: h_iDBEFS_4(csize, in, out); break;
          case DBEFS_8: h_iDBEFS_8(csize, in, out); break;
          case DBESF_4: h_iDBESF_4(csize, in, out); break;
          case DBESF_8: h_iDBESF_8(csize, in, out); break;
          case DIFFMS_1: h_iDIFFMS_1(csize, in, out); break;
          case DIFFMS_2: h_iDIFFMS_2(csize, in, out); break;
          case DIFFMS_4: h_iDIFFMS_4(csize, in, out); break;
          case DIFFMS_8: h_iDIFFMS_8(csize, in, out); break;
          case DIFFNB_1: h_iDIFFNB_1(csize, in, out); break;
          case DIFFNB_2: h_iDIFFNB_2(csize, in, out); break;
          case DIFFNB_4: h_iDIFFNB_4(csize, in, out); break;
          case DIFFNB_8: h_iDIFFNB_8(csize, in, out); break;
          case DIFF_1: h_iDIFF_1(csize, in, out); break;
          case DIFF_2: h_iDIFF_2(csize, in, out); break;
          case DIFF_4: h_iDIFF_4(csize, in, out); break;
          case DIFF_8: h_iDIFF_8(csize, in, out); break;
          case HANS_1: h_iHANS_1(csize, in, out); break;
          case HANS_2: h_iHANS_2(csize, in, out); break;
          case HCLOG_1: h_iHCLOG_1(csize, in, out); break;
          case HCLOG_2: h_iHCLOG_2(csize, in, out); break;
          case HCLOG_4: h_iHCLOG_4(csize, in, out); break;
          case HCLOG_8: h_iHCLOG_8(csize, in, out); break;
          case RARE_1: h_iRARE_1(csize, in, out); break;
          case RARE_2: h_iRARE_2(csize, in, out); break;
          case RARE_4: h_iRARE_4(csize, in, out); break;
          case RARE_8: h_iRARE_8(csize, in, out); break;
          case RAZE_1: h_iRAZE_1(csize, in, out); break;
          case RAZE_2: h_iRAZE_2(csize, in, out); break;
          case RAZE_4: h_iRAZE_4(csize, in, out); break;
          case RAZE_8: h_iRAZE_8(csize, in, out); break;
          case RLE_1: h_iRLE_1(csize, in, out); break;
          case RLE_2: h_iRLE_2(csize, in, out); break;
          case RLE_4: h_iRLE_4(csize, in, out); break;
          case RLE_8: h_iRLE_8(csize, in, out); break;
          case RRE_1: h_iRRE_1(csize, in, out); break;
          case RRE_2: h_iRRE_2(csize, in, out); break;
          case RRE_4: h_iRRE_4(csize, in, out); break;
          case RRE_8: h_iRRE_8(csize, in, out); break;
          case RZE_1: h_iRZE_1(csize, in, out); break;
          case RZE_2: h_iRZE_2(csize, in, out); break;
          case RZE_4: h_iRZE_4(csize, in, out); break;
          case RZE_8: h_iRZE_8(csize, in, out); break;
          case TCMS_1: h_iTCMS_1(csize, in, out); break;
          case TCMS_2: h_iTCMS_2(csize, in, out); break;
          case TCMS_4: h_iTCMS_4(csize, in, out); break;
          case TCMS_8: h_iTCMS_8(csize, in, out); break;
          case TCNB_1: h_iTCNB_1(csize, in, out); break;
          case TCNB_2: h_iTCNB_2(csize, in, out); break;
          case TCNB_4: h_iTCNB_4(csize, in, out); break;
          case TCNB_8: h_iTCNB_8(csize, in, out); break;
          case TUPL12_1: h_iTUPL12_1(csize, in, out); break;
          case TUPL2_1: h_iTUPL2_1(csize, in, out); break;
          case TUPL2_2: h_iTUPL2_2(csize, in, out); break;
          case TUPL2_4: h_iTUPL2_4(csize, in, out); break;
          case TUPL3_1: h_iTUPL3_1(csize, in, out); break;
          case TUPL3_2: h_iTUPL3_2(csize, in, out); break;
          case TUPL3_8: h_iTUPL3_8(csize, in, out); break;
          case TUPL4_1: h_iTUPL4_1(csize, in, out); break;
          case TUPL4_2: h_iTUPL4_2(csize, in, out); break;
          case TUPL6_1: h_iTUPL6_1(csize, in, out); break;
          case TUPL6_2: h_iTUPL6_2(csize, in, out); break;
          case TUPL6_4: h_iTUPL6_4(csize, in, out); break;
          case TUPL6_8: h_iTUPL6_8(csize, in, out); break;
          case TUPL8_1: h_iTUPL8_1(csize, in, out); break;
          /*##switch-host-decode-end##*/
        }
        pipeline <<= 8;
      }
      if (csize != osize) {fprintf(stderr, "ERROR: csize %d does not match osize %d in chunk %lld\n\n", csize, osize, chunkID); throw std::runtime_error("LC error");}
      memcpy(&output[base], out, csize);
    }
  }

  // finish
  delete [] start;
}


static void h_decode(const unsigned long long chain, const byte* const __restrict__ input, byte* const __restrict__ output, long long& outsize, const int n_threads)
{
  #ifdef _OPENMP
  const int before = omp_get_max_threads();
  omp_set_num_threads(n_threads);
  #endif

  h_decode(chain, input, output, outsize);

  #ifdef _OPENMP
  omp_set_num_threads(before);
  #endif
}
#endif


#ifdef USE_GPU
struct GPUTimer
{
  cudaEvent_t beg, end;
  GPUTimer() {cudaEventCreate(&beg); cudaEventCreate(&end);}
  ~GPUTimer() {cudaEventDestroy(beg); cudaEventDestroy(end);}
  void start() {cudaEventRecord(beg, 0);}
  double stop() {cudaEventRecord(end, 0); cudaEventSynchronize(end); float ms; cudaEventElapsedTime(&ms, beg, end); return 0.001 * ms;}
};


static void CheckCuda(const int line)
{
  cudaError_t e;
  cudaDeviceSynchronize();
  if (cudaSuccess != (e = cudaGetLastError())) {
    fprintf(stderr, "CUDA error %d on line %d: %s\n\n", e, line, cudaGetErrorString(e));
    throw std::runtime_error("LC error");
  }
}
#endif


#ifdef USE_CPU
struct CPUTimer
{
  timeval beg, end;
  CPUTimer() {}
  ~CPUTimer() {}
  void start() {gettimeofday(&beg, NULL);}
  double stop() {gettimeofday(&end, NULL); return end.tv_sec - beg.tv_sec + (end.tv_usec - beg.tv_usec) / 1000000.0;}
};
#endif


static std::string getPipeline(unsigned long long pipeline, const int stages)
{
  std::string s;
  for (int i = 0; i < stages; i++) {
    switch (pipeline & 0xff) {
      default: s += " NUL"; break;
      /*##switch-pipeline-beg##*/
      case BIT_1: s += " BIT_1"; break;
      case BIT_2: s += " BIT_2"; break;
      case BIT_4: s += " BIT_4"; break;
      case BIT_8: s += " BIT_8"; break;
      case CLOG_1: s += " CLOG_1"; break;
      case CLOG_2: s += " CLOG_2"; break;
      case CLOG_4: s += " CLOG_4"; break;
      case CLOG_8: s += " CLOG_8"; break;
      case DBEFS_4: s += " DBEFS_4"; break;
      case DBEFS_8: s += " DBEFS_8"; break;
      case DBESF_4: s += " DBESF_4"; break;
      case DBESF_8: s += " DBESF_8"; break;
      case DIFFMS_1: s += " DIFFMS_1"; break;
      case DIFFMS_2: s += " DIFFMS_2"; break;
      case DIFFMS_4: s += " DIFFMS_4"; break;
      case DIFFMS_8: s += " DIFFMS_8"; break;
      case DIFFNB_1: s += " DIFFNB_1"; break;
      case DIFFNB_2: s += " DIFFNB_2"; break;
      case DIFFNB_4: s += " DIFFNB_4"; break;
      case DIFFNB_8: s += " DIFFNB_8"; break;
      case DIFF_1: s += " DIFF_1"; break;
      case DIFF_2: s += " DIFF_2"; break;
      case DIFF_4: s += " DIFF_4"; break;
      case DIFF_8: s += " DIFF_8"; break;
      case HANS_1: s += " HANS_1"; break;
      case HANS_2: s += " HANS_2"; break;
      case HANS_common: s += " HANS_common"; break;
      case HCLOG_1: s += " HCLOG_1"; break;
      case HCLOG_2: s += " HCLOG_2"; break;
      case HCLOG_4: s += " HCLOG_4"; break;
      case HCLOG_8: s += " HCLOG_8"; break;
      case RARE_1: s += " RARE_1"; break;
      case RARE_2: s += " RARE_2"; break;
      case RARE_4: s += " RARE_4"; break;
      case RARE_8: s += " RARE_8"; break;
      case RAZE_1: s += " RAZE_1"; break;
      case RAZE_2: s += " RAZE_2"; break;
      case RAZE_4: s += " RAZE_4"; break;
      case RAZE_8: s += " RAZE_8"; break;
      case RLE_1: s += " RLE_1"; break;
      case RLE_2: s += " RLE_2"; break;
      case RLE_4: s += " RLE_4"; break;
      case RLE_8: s += " RLE_8"; break;
      case RRE_1: s += " RRE_1"; break;
      case RRE_2: s += " RRE_2"; break;
      case RRE_4: s += " RRE_4"; break;
      case RRE_8: s += " RRE_8"; break;
      case RZE_1: s += " RZE_1"; break;
      case RZE_2: s += " RZE_2"; break;
      case RZE_4: s += " RZE_4"; break;
      case RZE_8: s += " RZE_8"; break;
      case TCMS_1: s += " TCMS_1"; break;
      case TCMS_2: s += " TCMS_2"; break;
      case TCMS_4: s += " TCMS_4"; break;
      case TCMS_8: s += " TCMS_8"; break;
      case TCNB_1: s += " TCNB_1"; break;
      case TCNB_2: s += " TCNB_2"; break;
      case TCNB_4: s += " TCNB_4"; break;
      case TCNB_8: s += " TCNB_8"; break;
      case TUPL12_1: s += " TUPL12_1"; break;
      case TUPL2_1: s += " TUPL2_1"; break;
      case TUPL2_2: s += " TUPL2_2"; break;
      case TUPL2_4: s += " TUPL2_4"; break;
      case TUPL3_1: s += " TUPL3_1"; break;
      case TUPL3_2: s += " TUPL3_2"; break;
      case TUPL3_8: s += " TUPL3_8"; break;
      case TUPL4_1: s += " TUPL4_1"; break;
      case TUPL4_2: s += " TUPL4_2"; break;
      case TUPL6_1: s += " TUPL6_1"; break;
      case TUPL6_2: s += " TUPL6_2"; break;
      case TUPL6_4: s += " TUPL6_4"; break;
      case TUPL6_8: s += " TUPL6_8"; break;
      case TUPL8_1: s += " TUPL8_1"; break;
      /*##switch-pipeline-end##*/
    }
    pipeline >>= 8;
  }
  s.erase(0, 1);
  return s;
}


static std::map<std::string, byte> getPreproMap()
{
  std::map<std::string, byte> preprocessors;
  preprocessors["NUL"] = 0;
  /*##preprocessor-map-beg##*/
  preprocessors["QUANT_NOA_0_f64"] = QUANT_NOA_0_f64;
  preprocessors["LOR1D_i32"] = LOR1D_i32;
  preprocessors["QUANT_NOA_R_f32"] = QUANT_NOA_R_f32;
  preprocessors["QUANT_NOA_0_f32"] = QUANT_NOA_0_f32;
  preprocessors["QUANT_ABS_0_f32"] = QUANT_ABS_0_f32;
  preprocessors["QUANT_REL_0_f32"] = QUANT_REL_0_f32;
  preprocessors["QUANT_REL_R_f64"] = QUANT_REL_R_f64;
  preprocessors["QUANT_ABS_0_f64"] = QUANT_ABS_0_f64;
  preprocessors["QUANT_REL_R_f32"] = QUANT_REL_R_f32;
  preprocessors["QUANT_ABS_R_f32"] = QUANT_ABS_R_f32;
  preprocessors["QUANT_NOA_R_f64"] = QUANT_NOA_R_f64;
  preprocessors["QUANT_ABS_R_f64"] = QUANT_ABS_R_f64;
  preprocessors["QUANT_REL_0_f64"] = QUANT_REL_0_f64;
  /*##preprocessor-map-end##*/
  return preprocessors;
}


static std::string getPreprocessors(std::vector<std::pair<byte, std::vector<double>>> prepros)
{
  std::string s;

  if (prepros.size() > 0) {
    const std::map<std::string, byte> prepro_name2num = getPreproMap();
    std::string prepro_num2name [256];
    for (auto pair: prepro_name2num) {
      prepro_num2name[pair.second] = pair.first;
    }

    for (int i = 0; i < prepros.size(); i++) {
      s += ' ';
      s += prepro_num2name[prepros[i].first];
      s += '(';
      bool first = true;
      for (double d: prepros[i].second) {
        if (first) {
          first = false;
        } else {
          s += ", ";
        }
        long long val = d;
        if (d == val) {
          s += std::to_string(val);
        } else {
          s += std::to_string(d);
        }
      }
      s += ')';
    }

    s.erase(0, 1);
  }

  return s;
}


static void printPreprocessors(FILE* f = stdout)
{
  const std::map<std::string, byte> prepro_name2num = getPreproMap();
  if (f == stdout) {
    fprintf(f, "%ld available preprocessors:\n", prepro_name2num.size());
    for (auto pair: prepro_name2num) {
      fprintf(f, "%s ", pair.first.c_str());
    }
    fprintf(f, "\n");
  } else {
    fprintf(f, "available preprocessors, %ld\n", prepro_name2num.size());
    for (auto pair: prepro_name2num) {
      fprintf(f, "%s\n", pair.first.c_str());
    }
  }
}


static std::map<std::string, byte> getCompMap()
{
  std::map<std::string, byte> components;
  components["NUL"] = 0;
  /*##component-map-beg##*/
  components["BIT_1"] = 1;
  components["BIT_2"] = 2;
  components["BIT_4"] = 3;
  components["BIT_8"] = 4;
  components["CLOG_1"] = 5;
  components["CLOG_2"] = 6;
  components["CLOG_4"] = 7;
  components["CLOG_8"] = 8;
  components["DBEFS_4"] = 9;
  components["DBEFS_8"] = 10;
  components["DBESF_4"] = 11;
  components["DBESF_8"] = 12;
  components["DIFFMS_1"] = 13;
  components["DIFFMS_2"] = 14;
  components["DIFFMS_4"] = 15;
  components["DIFFMS_8"] = 16;
  components["DIFFNB_1"] = 17;
  components["DIFFNB_2"] = 18;
  components["DIFFNB_4"] = 19;
  components["DIFFNB_8"] = 20;
  components["DIFF_1"] = 21;
  components["DIFF_2"] = 22;
  components["DIFF_4"] = 23;
  components["DIFF_8"] = 24;
  components["HANS_1"] = 25;
  components["HANS_2"] = 26;
  components["HANS_common"] = 27;
  components["HCLOG_1"] = 28;
  components["HCLOG_2"] = 29;
  components["HCLOG_4"] = 30;
  components["HCLOG_8"] = 31;
  components["RARE_1"] = 32;
  components["RARE_2"] = 33;
  components["RARE_4"] = 34;
  components["RARE_8"] = 35;
  components["RAZE_1"] = 36;
  components["RAZE_2"] = 37;
  components["RAZE_4"] = 38;
  components["RAZE_8"] = 39;
  components["RLE_1"] = 40;
  components["RLE_2"] = 41;
  components["RLE_4"] = 42;
  components["RLE_8"] = 43;
  components["RRE_1"] = 44;
  components["RRE_2"] = 45;
  components["RRE_4"] = 46;
  components["RRE_8"] = 47;
  components["RZE_1"] = 48;
  components["RZE_2"] = 49;
  components["RZE_4"] = 50;
  components["RZE_8"] = 51;
  components["TCMS_1"] = 52;
  components["TCMS_2"] = 53;
  components["TCMS_4"] = 54;
  components["TCMS_8"] = 55;
  components["TCNB_1"] = 56;
  components["TCNB_2"] = 57;
  components["TCNB_4"] = 58;
  components["TCNB_8"] = 59;
  components["TUPL12_1"] = 60;
  components["TUPL2_1"] = 61;
  components["TUPL2_2"] = 62;
  components["TUPL2_4"] = 63;
  components["TUPL3_1"] = 64;
  components["TUPL3_2"] = 65;
  components["TUPL3_8"] = 66;
  components["TUPL4_1"] = 67;
  components["TUPL4_2"] = 68;
  components["TUPL6_1"] = 69;
  components["TUPL6_2"] = 70;
  components["TUPL6_4"] = 71;
  components["TUPL6_8"] = 72;
  components["TUPL8_1"] = 73;
  /*##component-map-end##*/
  return components;
}


static void printComponents(FILE* f = stdout)
{
  const std::map<std::string, byte> comp_name2num = getCompMap();
  if (f == stdout) {
    fprintf(f, "%ld available components:\n", comp_name2num.size());
    for (auto pair: comp_name2num) {
      fprintf(f, "%s ", pair.first.c_str());
    }
    fprintf(f, "\n");
  } else {
    fprintf(f, "available components, %ld\n", comp_name2num.size());
    for (auto pair: comp_name2num) {
      fprintf(f, "%s\n", pair.first.c_str());
    }
  }
}


static std::map<std::string, byte> getVerifMap()
{
  std::map<std::string, byte> verifs;
  /*##verifier-map-beg##*/
  verifs["MAXNOA_f32"] = v_MAXNOA_f32;
  verifs["PSNR_f32"] = v_PSNR_f32;
  verifs["MAXABS_f64"] = v_MAXABS_f64;
  verifs["MSE_f64"] = v_MSE_f64;
  verifs["PSNR_f64"] = v_PSNR_f64;
  verifs["MAXNOA_f64"] = v_MAXNOA_f64;
  verifs["MAXREL_f32"] = v_MAXREL_f32;
  verifs["MAXABS_f32"] = v_MAXABS_f32;
  verifs["PASS"] = v_PASS;
  verifs["MSE_f32"] = v_MSE_f32;
  verifs["LOSSLESS"] = v_LOSSLESS;
  verifs["MAXREL_f64"] = v_MAXREL_f64;
  /*##verifier-map-end##*/
  return verifs;
}


template <typename T>
static double Entropy(T *const data, const long long len)
{
  assert(sizeof(T) <= 2);
  const int size = 1 << (sizeof(T) * 8);
  long long hist [size];
  memset(hist, 0, size * sizeof(long long));
  for (long long i = 0; i < len; i++) {
    hist[data[i]]++;
  }
  double invtot = 1.0 / len;
  double sum = 0.0;
  for (int i = 0; i < size; i++) {
    if (hist[i] != 0) {
      double ent = hist[i] * invtot;
      sum += ent * log2(ent);
    }
  }
  return -sum;
}


template <typename T>
static double entropy(const T* const data, const long long len)
{
  double sum = 0.0;
  if (len > 0) {
    T* const copy = new T [len];
    for (long long i = 0; i < len; i++) copy[i] = data[i];
    std::sort(&copy[0], &copy[len]);

    const double invlen = 1.0 / len;
    long long cnt = 1;
    T prev = copy[0];
    for (long long i = 1; i < len; i++) {
      if (copy[i] == prev) {
        cnt++;
      } else {
        const double ent = cnt * invlen;
        sum += ent * log2(ent);
        cnt = 1;
        prev = copy[i];
      }
    }
    const double ent = cnt * invlen;
    sum += ent * log2(ent);
    sum = -sum;
    delete [] copy;
  }
  return sum;
}


template <typename T>
static void Frequency(const T* const data, const long long len)
{
  assert(sizeof(T) <= 2);
  const int size = 1 << (sizeof(T) * 8);
  long long hist [size];
  memset(hist, 0, size * sizeof(long long));
  for (long long i = 0; i < len; i++) {
    hist[data[i]]++;
  }
  std::vector<std::pair<long long, T>> vec;
  for (int i = 0; i < size; i++) {
    if (hist[i] != 0) {
      vec.push_back(std::make_pair(-hist[i], (T)i));
    }
  }

  printf(" unique values: %ld\n", vec.size());
  printf(" occurrences\n");
  std::sort(vec.begin(), vec.end());
  for (int i = 0; i < std::min(8, (int)vec.size()); i++) {
    printf(" %14lld: %14lld  (%6.3f%%)\n", (long long)vec[i].second, -vec[i].first, -100.0 * vec[i].first / len);
  }
}


template <typename T>
static void frequency(const T* const data, const long long len)
{
  std::vector<std::pair<int, T>> vec;
  if (len > 0) {
    T* const copy = new T [len];
    for (long long i = 0; i < len; i++) copy[i] = data[i];
    std::sort(&copy[0], &copy[len]);

    int cnt = 1;
    T prev = copy[0];
    for (long long i = 1; i < len; i++) {
      if (copy[i] == prev) {
        cnt++;
      } else {
        vec.push_back(std::make_pair(-cnt, prev));
        cnt = 1;
        prev = copy[i];
      }
    }
    vec.push_back(std::make_pair(-cnt, prev));
    delete [] copy;
  }

  printf(" unique values: %ld\n", vec.size());
  printf(" occurrences\n");
  std::sort(vec.begin(), vec.end());
  for (int i = 0; i < std::min(8, (int)vec.size()); i++) {
    printf(" %20lld: %20d  (%6.3f%%)\n", (long long)vec[i].second, -vec[i].first, -100.0 * vec[i].first / len);
  }
}


struct Elem {
  unsigned long long pipe;
  float CR;
  float HencThru;
  float HdecThru;
  float DencThru;
  float DdecThru;
};


#ifdef USE_GPU
static bool compareElemDencThru(Elem e1, Elem e2)
{
  return (e1.CR < e2.CR) || ((e1.CR == e2.CR) && (e1.DencThru < e2.DencThru));
}


static bool compareElemDdecThru(Elem e1, Elem e2)
{
  return (e1.CR < e2.CR) || ((e1.CR == e2.CR) && (e1.DdecThru < e2.DdecThru));
}
#endif


#ifdef USE_CPU
static bool compareElemHencThru(Elem e1, Elem e2)
{
  return (e1.CR < e2.CR) || ((e1.CR == e2.CR) && (e1.HencThru < e2.HencThru));
}


static bool compareElemHdecThru(Elem e1, Elem e2)
{
  return (e1.CR < e2.CR) || ((e1.CR == e2.CR) && (e1.HdecThru < e2.HdecThru));
}
#endif


static std::vector<std::pair<byte, std::vector<double>>> getItems(std::map<std::string, byte> item_name2num, char* const s)
{
  std::vector<std::pair<byte, std::vector<double>>> items;

  char* p = s;
  while ((*p != 0) && ((*p == ' ') || (*p == '\t'))) p++;  // skip over white space
  while (*p != 0) {
    // get name
    char* beg = p;
    while ((*p != 0) && (*p != ' ') && (*p != '\t') && (*p != '(')) p++;  // find end of name
    char* end = p;
    if (end <= beg) {fprintf(stderr, "ERROR: expected an item name in specification\n\n"); throw std::runtime_error("LC error");}
    char old = *end;
    *end = 0;  // string terminator
    int num = -1;
    for (auto pair: item_name2num) {
      const std::string itemname = pair.first;
      const byte itemnum = pair.second;
      if (itemname.compare(beg) == 0) {
        num = itemnum;
        break;
      }
    }
    if (num < 0) {fprintf(stderr, "ERROR: unknown item name\n\n"); throw std::runtime_error("LC error");}
    *end = old;

    // read in parameters
    std::vector<double> params;
    while ((*p != 0) && ((*p == ' ') || (*p == '\t'))) p++;  // skip over white space
    if (*p != '(') {fprintf(stderr, "ERROR: expected '(' in specification\n\n"); throw std::runtime_error("LC error");}
    p++;
    while ((*p != 0) && ((*p == ' ') || (*p == '\t'))) p++;  // skip over white space
    while ((*p != 0) && (*p != ')')) {
      // get double
      char* pos;
      const double d = std::strtod(p, &pos);
      if (pos == p) {fprintf(stderr, "ERROR: expected a value in specification\n\n"); throw std::runtime_error("LC error");}
      p = pos;
      params.push_back(d);
      while ((*p != 0) && ((*p == ' ') || (*p == '\t'))) p++;  // skip over white space
      if (*p == ')') break;

      // consume comma
      if (*p != ',') {fprintf(stderr, "ERROR: expected ',' in specification\n\n"); throw std::runtime_error("LC error");}
      p++;
      while ((*p != 0) && ((*p == ' ') || (*p == '\t'))) p++;  // skip over white space
    }
    if (*p != ')') {fprintf(stderr, "ERROR: expected ')' in specification\n\n"); throw std::runtime_error("LC error");}
    p++;
    items.push_back(std::make_pair((byte)num, params));
    while ((*p != 0) && ((*p == ' ') || (*p == '\t'))) p++;  // skip over white space
  }

  return items;
}


static std::vector<std::vector<byte>> getStages(std::map<std::string, byte> comp_name2num, char* const regex, int& stages, unsigned long long& algorithms)
{
  std::vector<std::vector<byte>> comp_list;

  int s = 0;
  char* ptr = strtok(regex, " \t");
  while (ptr != NULL) {
    if (s >= max_stages) {fprintf(stderr, "ERROR: number of stages must be between 1 and %d\n\n", max_stages); throw std::runtime_error("LC error");}

    std::vector<byte> list;
    std::string in = ptr;
    const bool inv = (in[0] == '~');
    if (inv) in = in.substr(1);
    const std::regex re(in);
    for (auto pair: comp_name2num) {
      const std::string compname = pair.first;
      const byte compnum = pair.second;
      if (std::regex_match(compname, re)) {
        if (!inv) list.push_back(compnum);
      } else {
        if (inv) list.push_back(compnum);
      }
    }
    comp_list.push_back(list);
    s++;
    ptr = strtok(NULL, " \t");
  }

  stages = s;
  if (stages < 1) {fprintf(stderr, "ERROR: stages must be between 1 and %d\n\n", max_stages); throw std::runtime_error("LC error");}

  algorithms = 1;
  for (int s = 0; s < stages; s++) {
    algorithms *= comp_list[s].size();
  }

  return comp_list;
}


static void printStages(std::vector<std::pair<byte, std::vector<double>>> prepros, std::map<std::string, byte> prepro_name2num, std::vector<std::vector<byte>> comp_list, std::map<std::string, byte> comp_name2num, const int stages, const unsigned long long algorithms, FILE* f = stdout)
{
  std::string prepro_num2name [256];
  for (auto pair: prepro_name2num) {
    prepro_num2name[pair.second] = pair.first;
  }

  std::string comp_num2name [256];
  for (auto pair: comp_name2num) {
    comp_num2name[pair.second] = pair.first;
  }

  int max = 0;
  for (int s = 0; s < stages; s++) {
    max = std::max(max, (int)comp_list[s].size());
  }

  if (f == stdout) {
    printf("algorithms: %lld\n\n", algorithms);
    if (prepros.size() > 0) {
      printf("  preprocessors\n  -------------\n");
      for (int i = 0; i < prepros.size(); i++) {
        printf("  %s(", prepro_num2name[prepros[i].first].c_str());
        bool first = true;
        for (double d: prepros[i].second) {
          if (first) {
            first = false;
          } else {
            printf(", ");
          }
          long long val = d;
          if (d == val) {
            printf("%lld", val);
          } else {
            printf("%e", d);
          }
        }
        printf(")");
      }
      printf("\n\n");
    }

    for (int s = 0; s < stages; s++) printf("  stage %d", s + 1);
    printf("\n");
    for (int s = 0; s < stages; s++) printf("  -------");
    printf("\n");
    for (int e = 0; e < max; e++) {
      for (int s = 0; s < stages; s++) {
        if (e < comp_list[s].size()) {
          printf("%9s", comp_num2name[comp_list[s][e]].c_str());
        } else {
          printf("%9s", "");
        }
      }
      printf("\n");
    }
    printf("\n");
  } else {
    fprintf(f, "algorithms, %lld\n\n", algorithms);
    if (prepros.size() > 0) {
      fprintf(f, "preprocessors\n");
      for (int i = 0; i < prepros.size(); i++) {
        fprintf(f, "%s", prepro_num2name[prepros[i].first].c_str());
        for (double d: prepros[i].second) {
          long long val = d;
          if (d == val) {
            fprintf(f, ", %lld", val);
          } else {
            fprintf(f, ", %e", d);
          }
        }
        fprintf(f, "\n");
      }
      fprintf(f, "\n");
    }

    for (int s = 0; s < stages; s++) fprintf(f, "stage %d, ", s + 1);
    fprintf(f, "\n");
    for (int e = 0; e < max; e++) {
      for (int s = 0; s < stages; s++) {
        if (e < comp_list[s].size()) {
          fprintf(f, "%s, ", comp_num2name[comp_list[s][e]].c_str());
        } else {
          fprintf(f, ", ");
        }
      }
      fprintf(f, "\n");
    }
    fprintf(f, "\n");
  }
}


static void printUsage(char* argv [])
{
  printf("USAGE: %s input_file_name AL \"[preprocessor_name ...]\" \"component_name_regex [component_name_regex ...]\" [\"verifier\"]\n", argv[0]);
  printf("USAGE: %s input_file_name PR \"[preprocessor_name ...]\" \"component_name_regex [component_name_regex ...]\" [\"verifier\"]\n", argv[0]);
  printf("USAGE: %s input_file_name CR \"[preprocessor_name ...]\" \"component_name_regex [component_name_regex ...]\"\n", argv[0]);
  printf("USAGE: %s input_file_name EX \"[preprocessor_name ...]\" \"component_name_regex [component_name_regex ...]\" [\"verifier\"]\n", argv[0]);
  printf("USAGE: %s input_file_name TS\n", argv[0]);
  printf("\n");
  printPreprocessors();
  printf("\n");
  printComponents();
  printf("\nFor usage examples, please see the quick-start guide and tutorial at https://github.com/burtscher/LC-framework/.\n");
/*
  printf("\nExamples:\n");
  printf("1. Lossless 2-stage pipeline with CLOG in 2nd stage using 4-byte granularity, showing only compression ratio (CR):\n\n   ./lc input_file_name CR \"\" \".+ CLOG_4\"\n\n");
  printf("2. Lossless 3-stage pipeline with 3D Lorenzo preprocessor and a DIFF, open, and R2E stage using 8-byte granularity, showing compression ratio, compression and decompression throughput, and Pareto frontier (EX):\n\n   ./lc input_file_name EX \"LOR3D_i32(dim1, dim2, dim3)\" \"DIFF_8 .+ R2E_8\"\n\n");
  printf("3. Lossy 2-stage pipeline with quantization with a 0.001 error bound and a limit value of 1000 and a CLOG component using 4-byte granularity, showing only compression ratio:\n\n  ./lc input_file_name CR \"QUANT_ABS_R_f32(0.001, 1000)\" \"CLOG_4\"\n\n");
  printf("Notes:\n1. The double quotations are always needed, even if there is nothing between them.\n2. The Lorenzo preprocessors only work when the passed dimensions match the input file size.\n");
  printf("3. The quantization preprocessors always need an error bound parameter specified in parentheses.\n");
  printf("4. The quantization preprocessors optionally take a second value, which indicates the absolute value beyond which the values are compressed losslessly.\n");
  printf("5. See the ./verifiers directory for a list of available verifiers. Verifiers take an error bound as parameter.\n\n");
*/
}


struct Config {
  bool speed;  // print speed info
  bool size;  // print size info
  bool warmup;  // perform warmup run
  bool memcopy;  // measure memcopy speed
  bool decom;  // perform decompression
  bool verify;  // verify results
  bool csv;  // output CSV file
};


#endif  /* LC_FRAMEWORK_COMMON_H */
