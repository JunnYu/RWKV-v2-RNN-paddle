#include "paddle/extension.h"
#include <vector>
#include <stdio.h>

// require T <= Tmax, T % 4 == 0, B % BF == 0, B % BB === 0 (Tmax and BF and BB are passed by compiler)

#define F4(A, B) ((float4 *)(A))[(B) >> 2]

template <typename data_t>
__global__ void timex_cuda_forward_kernel(const data_t *__restrict__ const __w, const data_t *__restrict__ const __k, data_t *__restrict__ const x,
                               const data_t eps, const int B, const int C, const int T) {
    const int i = blockIdx.y;
    const int ij = (B * C) / BF;
    const int t = threadIdx.x << 2;

    __shared__ data_t ww[Tmax];
    __shared__ data_t kk[Tmax * BF];
    F4(ww, t) = F4(__w, t + T * (i % C));
    
    #pragma unroll
    for (int j = 0; j < BF; j++) {
        F4(kk, t + Tmax * j) = F4(__k, t + T * (i + ij * j));
    }
    __syncthreads();

    float4 s[BF];
    #pragma unroll
    for (int j = 0; j < BF; j++) {
        s[j] = {eps, eps, eps, eps};
    }
    const data_t *__restrict__ const w = ww + T - t - 4;
    for (int u = 0; u <= t; u++) {
        #pragma unroll
        for (int j = 0; j < BF; j++) {
            const data_t x = kk[u + Tmax * j];
            s[j].x += w[u + 3] * x;
            s[j].y += w[u + 2] * x;
            s[j].z += w[u + 1] * x;
            s[j].w += w[u + 0] * x;
        }
    }
    #pragma unroll
    for (int j = 0; j < BF; j++) {
        const data_t *__restrict__ const k = kk + Tmax * j;
        s[j].y += w[t + 3] * k[t + 1];
        s[j].z += w[t + 2] * k[t + 1];
        s[j].z += w[t + 3] * k[t + 2];
        s[j].w += w[t + 1] * k[t + 1];
        s[j].w += w[t + 2] * k[t + 2];
        s[j].w += w[t + 3] * k[t + 3];
        F4(x, t + T * (i + ij * j)) = s[j];
    }
}

std::vector<paddle::Tensor> timex_cuda_forward(const paddle::Tensor& w, const paddle::Tensor& k, float eps, int B, int C, int T) {
    dim3 gridDim(1, B * C / 4);
    dim3 blockDim(T >> 2);
    auto kw = paddle::Tensor(paddle::PlaceType::kGPU, k.shape());

    // call forward_kernel
    PD_DISPATCH_FLOATING_TYPES(
        w.type(), "timex_cuda_forward_kernel", ([&] {
            timex_cuda_forward_kernel<data_t><<<gridDim, blockDim>>>(
                w.data<data_t>(),
                k.data<data_t>(),
                kw.mutable_data<data_t>(k.place()),
                eps, B, C, T);
        }));
  
    return {kw};
}

template <typename data_t>
__global__ void timex_cuda_backward_kernel(const data_t *__restrict__ const __w, const data_t *__restrict__ const __k, const data_t *__restrict__ const __gwk,
                                data_t *__restrict__ const gw, data_t *__restrict__ const gk,
                                const int B, const int C, const int T) {
    const int i = blockIdx.y;
    const int ij = (B * C) / BB;
    const int t = threadIdx.x << 2;

    __shared__ data_t w[Tmax];
    __shared__ data_t kk[Tmax * BB];
    __shared__ data_t gg[Tmax * BB];
    F4(w, t) = F4(__w, t + T * (i % C));

    #pragma unroll
    for (int j = 0; j < BB; j++) {
        F4(kk, t + Tmax * j) = F4(__k, t + T * (i + ij * j));
        F4(gg, t + Tmax * j) = F4(__gwk, t + T * (i + ij * j));
    }
    __syncthreads();

    float4 s[BB];
    #pragma unroll
    for (int j = 0; j < BB; j++) {
        s[j] = {0, 0, 0, 0};
    }

    for (int u = 0; u <= t; u++) {
        #pragma unroll
        for (int j = 0; j < BB; j++) {
            const data_t *__restrict__ const g = gg + Tmax * j + T - t - 4;
            data_t x = kk[u + Tmax * j];
            s[j].x += g[u + 3] * x;
            s[j].y += g[u + 2] * x;
            s[j].z += g[u + 1] * x;
            s[j].w += g[u + 0] * x;
        }
    }
    #pragma unroll
    for (int j = 0; j < BB; j++) {
        const data_t *__restrict__ const k = kk + Tmax * j;
        const data_t *__restrict__ const g = gg + Tmax * j + T - t - 4;
        s[j].y += g[t + 3] * k[t + 1];
        s[j].z += g[t + 2] * k[t + 1];
        s[j].z += g[t + 3] * k[t + 2];
        s[j].w += g[t + 1] * k[t + 1];
        s[j].w += g[t + 2] * k[t + 2];
        s[j].w += g[t + 3] * k[t + 3];
        F4(gw, t + T * (i + ij * j)) = s[j];
    }

    #pragma unroll
    for (int j = 0; j < BB; j++) {
        s[j] = {0, 0, 0, 0};
    }

    for (int u = t + 3; u < T; u++) {
        data_t x = w[u];
        #pragma unroll
        for (int j = 0; j < BB; j++) {
            const data_t *__restrict__ const g = gg + Tmax * j + T + t - 3;
            s[j].x += g[2 - u] * x;
            s[j].y += g[3 - u] * x;
            s[j].z += g[4 - u] * x;
            s[j].w += g[5 - u] * x;
        }        
    }
    #pragma unroll
    for (int j = 0; j < BB; j++) {
        const data_t *__restrict__ const g = gg + Tmax * j + T + t - 3;
        s[j].x += g[2 - t] * w[t + 0];
        s[j].x += g[1 - t] * w[t + 1];
        s[j].x += g[0 - t] * w[t + 2];
        s[j].y += g[2 - t] * w[t + 1];
        s[j].y += g[1 - t] * w[t + 2];
        s[j].z += g[2 - t] * w[t + 2];
        F4(gk, t + T * (i + ij * j)) = s[j];
    }
}


std::vector<paddle::Tensor> timex_cuda_backward(const paddle::Tensor& w, const paddle::Tensor& k, const paddle::Tensor& gwk, int B, int C, int T) {
    dim3 gridDim(1, B * C / 2);
    dim3 blockDim(T >> 2);
    auto gw = paddle::Tensor(paddle::PlaceType::kGPU, k.shape());
    auto gk = paddle::Tensor(paddle::PlaceType::kGPU, k.shape());

    // call backward_kernel
    PD_DISPATCH_FLOATING_TYPES(
        w.type(), "timex_cuda_backward_kernel", ([&] {
            timex_cuda_backward_kernel<data_t><<<gridDim, blockDim>>>(
                w.data<data_t>(),
                k.data<data_t>(),
                gwk.data<data_t>(),
                gw.mutable_data<data_t>(k.place()),
                gk.mutable_data<data_t>(k.place()),
                B, C, T);
        }));

    return {gw, gk};
}
