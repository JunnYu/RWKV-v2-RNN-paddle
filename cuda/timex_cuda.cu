#include "paddle/extension.h"
#include <vector>
#include <stdio.h>

// require T <= 1024, T % 4 == 0, B % 4 == 0, B % 2 === 0 (1024 and 4 and 2 are passed by compiler)

#define F4(A, B) ((float4 *)(A))[(B) >> 2]

template <typename F>
__global__ void timex_cuda_forward_kernel(const F *__restrict__ const __w, const F *__restrict__ const __k, F *__restrict__ const x,
                               const F eps, const int B, const int C, const int T) {
    const int i = blockIdx.y;
    const int ij = (B * C) / 4;
    const int t = threadIdx.x << 2;

    __shared__ F ww[1024];
    __shared__ F kk[1024 * 4];
    F4(ww, t) = F4(__w, t + T * (i % C));
    
    #pragma unroll
    for (int j = 0; j < 4; j++) {
        F4(kk, t + 1024 * j) = F4(__k, t + T * (i + ij * j));
    }
    __syncthreads();

    float4 s[4];
    #pragma unroll
    for (int j = 0; j < 4; j++) {
        s[j] = {eps, eps, eps, eps};
    }
    const F *__restrict__ const w = ww + T - t - 4;
    for (int u = 0; u <= t; u++) {
        #pragma unroll
        for (int j = 0; j < 4; j++) {
            const F x = kk[u + 1024 * j];
            s[j].x += w[u + 3] * x;
            s[j].y += w[u + 2] * x;
            s[j].z += w[u + 1] * x;
            s[j].w += w[u + 0] * x;
        }
    }
    #pragma unroll
    for (int j = 0; j < 4; j++) {
        const F *__restrict__ const k = kk + 1024 * j;
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
    auto out = paddle::Tensor(paddle::PlaceType::kGPU, k.shape());

    // call forward_kernel
    timex_cuda_forward_kernel<float><<<gridDim, blockDim>>>(
        w.data<float>(), k.data<float>(), out.mutable_data<float>(k.place()), eps, B, C, T);
  
    return {out};
}

template <typename F>
__global__ void timex_cuda_backward_kernel(const F *__restrict__ const __w, const F *__restrict__ const __k, const F *__restrict__ const __gwk,
                                F *__restrict__ const gw, F *__restrict__ const gk,
                                const int B, const int C, const int T) {
    const int i = blockIdx.y;
    const int ij = (B * C) / 2;
    const int t = threadIdx.x << 2;

    __shared__ F w[1024];
    __shared__ F kk[1024 * 2];
    __shared__ F gg[1024 * 2];
    F4(w, t) = F4(__w, t + T * (i % C));

    #pragma unroll
    for (int j = 0; j < 2; j++) {
        F4(kk, t + 1024 * j) = F4(__k, t + T * (i + ij * j));
        F4(gg, t + 1024 * j) = F4(__gwk, t + T * (i + ij * j));
    }
    __syncthreads();

    float4 s[2];
    #pragma unroll
    for (int j = 0; j < 2; j++) {
        s[j] = {0, 0, 0, 0};
    }

    for (int u = 0; u <= t; u++) {
        #pragma unroll
        for (int j = 0; j < 2; j++) {
            const F *__restrict__ const g = gg + 1024 * j + T - t - 4;
            F x = kk[u + 1024 * j];
            s[j].x += g[u + 3] * x;
            s[j].y += g[u + 2] * x;
            s[j].z += g[u + 1] * x;
            s[j].w += g[u + 0] * x;
        }
    }
    #pragma unroll
    for (int j = 0; j < 2; j++) {
        const F *__restrict__ const k = kk + 1024 * j;
        const F *__restrict__ const g = gg + 1024 * j + T - t - 4;
        s[j].y += g[t + 3] * k[t + 1];
        s[j].z += g[t + 2] * k[t + 1];
        s[j].z += g[t + 3] * k[t + 2];
        s[j].w += g[t + 1] * k[t + 1];
        s[j].w += g[t + 2] * k[t + 2];
        s[j].w += g[t + 3] * k[t + 3];
        F4(gw, t + T * (i + ij * j)) = s[j];
    }

    #pragma unroll
    for (int j = 0; j < 2; j++) {
        s[j] = {0, 0, 0, 0};
    }

    for (int u = t + 3; u < T; u++) {
        F x = w[u];
        #pragma unroll
        for (int j = 0; j < 2; j++) {
            const F *__restrict__ const g = gg + 1024 * j + T + t - 3;
            s[j].x += g[2 - u] * x;
            s[j].y += g[3 - u] * x;
            s[j].z += g[4 - u] * x;
            s[j].w += g[5 - u] * x;
        }        
    }
    #pragma unroll
    for (int j = 0; j < 2; j++) {
        const F *__restrict__ const g = gg + 1024 * j + T + t - 3;
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
    timex_cuda_backward_kernel<float><<<gridDim, blockDim>>>(
        w.data<float>(),
        k.data<float>(),
        gwk.data<float>(),
        gw.mutable_data<float>(k.place()),
        gk.mutable_data<float>(k.place()),
        B, C, T);
    
    return {gw, gk};
}
