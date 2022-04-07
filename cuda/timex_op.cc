#include "paddle/extension.h"
#include <vector>
#include <stdio.h>

std::vector<paddle::Tensor> timex_cuda_forward(const paddle::Tensor& w, const paddle::Tensor& k, float eps, int B, int C, int T);
std::vector<paddle::Tensor> timex_cuda_backward(const paddle::Tensor& w, const paddle::Tensor& k, const paddle::Tensor& gwk, int B, int C, int T);

std::vector<paddle::Tensor> TimeXForward(const paddle::Tensor& w, const paddle::Tensor& k, int B, int C, int T, float eps)
{
 
    if (w.place() == paddle::PlaceType::kGPU)
    {
        return timex_cuda_forward(w, k, eps, B, C, T);
    }
    else
    {
        PD_THROW("Not implemented.");
    }
}

std::vector<paddle::Tensor> TimeXBackward(const paddle::Tensor& w, const paddle::Tensor& k, const paddle::Tensor& gwk, int B, int C, int T)
{
    if (w.place() == paddle::PlaceType::kGPU)
    {
        return timex_cuda_backward(w, k, gwk, B, C, T);
    }
    else
    {
        PD_THROW("Not implemented.");
    }
}


std::vector<paddle::DataType> TimeXInferDtype(paddle::DataType x_dtype) {
  return {x_dtype};
}

PD_BUILD_OP(TimeX_forward)
    .Inputs({"W", "K"})
    .Outputs({"WK"})
    .Attrs({"B: int",
            "C: int",
            "T: int",
            "eps: float"})
    .SetKernelFn(PD_KERNEL(TimeXForward))
    .SetInferDtypeFn({PD_INFER_DTYPE(TimeXInferDtype)});

PD_BUILD_OP(TimeX_backward)
    .Inputs({"W", "K", "WK"})
    .Outputs({"GW", "GK"})
    .Attrs({"B: int",
            "C: int",
            "T: int"})
    .SetKernelFn(PD_KERNEL(TimeXBackward))
    .SetInferDtypeFn({PD_INFER_DTYPE(TimeXInferDtype)});

// PD_BUILD_GRAD_OP(custom_timex)
//     .Inputs({"W", "K", paddle::Grad("WK")})
//     .Outputs({paddle::Grad("W"), paddle::Grad("K")})
//     .Attrs({"B: int",
//             "C: int",
//             "T: int"})
//     .SetKernelFn(PD_KERNEL(TimeXBackward));