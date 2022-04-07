#include "paddle/extension.h"
#include <vector>
#include <stdio.h>

std::vector<paddle::Tensor> timex_cuda_forward(const paddle::Tensor &w, const paddle::Tensor &k, float eps, int B, int C, int T);
std::vector<paddle::Tensor> timex_cuda_backward(const paddle::Tensor &w, const paddle::Tensor &k, const paddle::Tensor &gwk, int B, int C, int T);

std::vector<paddle::Tensor> TimeXForward(const paddle::Tensor &w, const paddle::Tensor &k, int B, int C, int T, float eps)
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

std::vector<paddle::Tensor> TimeXBackward(const paddle::Tensor &w, const paddle::Tensor &k, const paddle::Tensor &gwk, int B, int C, int T)
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

// forward infer
std::vector<paddle::DataType> TimeXForwardInferDtype(paddle::DataType w_dtype, paddle::DataType k_dtype)
{
    return {w_dtype};
}

std::vector<std::vector<int64_t>> TimeXForwardInferShape(std::vector<int64_t> w_shape, std::vector<int64_t> k_shape)
{
    return {k_shape};
}

// backward infer
std::vector<paddle::DataType> TimeXBackwardInferDtype(paddle::DataType w_dtype, paddle::DataType k_dtype, paddle::DataType wk_dtype)
{
    return {w_dtype, k_dtype};
}

std::vector<std::vector<int64_t>> TimeXBackwardInferShape(std::vector<int64_t> w_shape, std::vector<int64_t> k_shape, std::vector<int64_t> wk_shape)
{
    return {k_shape, k_shape};
}

PD_BUILD_OP(forward)
    .Inputs({"W", "K"})
    .Outputs({"WK"})
    .Attrs({"B: int",
            "C: int",
            "T: int",
            "eps: float"})
    .SetInferShapeFn(PD_INFER_SHAPE(TimeXForwardInferShape))
    .SetKernelFn(PD_KERNEL(TimeXForward))
    .SetInferDtypeFn({PD_INFER_DTYPE(TimeXForwardInferDtype)});

PD_BUILD_OP(backward)
    .Inputs({"W", "K", "WK"})
    .Outputs({"GW", "GK"})
    .Attrs({"B: int",
            "C: int",
            "T: int"})
    .SetInferShapeFn(PD_INFER_SHAPE(TimeXBackwardInferShape))
    .SetKernelFn(PD_KERNEL(TimeXBackward))
    .SetInferDtypeFn({PD_INFER_DTYPE(TimeXBackwardInferDtype)});

// PD_BUILD_GRAD_OP(custom_timex)
//     .Inputs({"W", "K", paddle::Grad("WK")})
//     .Outputs({paddle::Grad("W"), paddle::Grad("K")})
//     .Attrs({"B: int",
//             "C: int",
//             "T: int"})
//     .SetKernelFn(PD_KERNEL(TimeXBackward));