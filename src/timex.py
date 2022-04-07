from paddle.autograd import PyLayer
from paddle.utils.cpp_extension import load

T_MAX = 1024  # increase this if your ctx_len > 1024
B_GROUP_FORWARD = 4  # set to 8 for best performance
B_GROUP_BACKWARD = 2  # set to 2 for best performance

timex_cuda = load(
    name="timex",
    sources=["src/cuda/timex_op.cc", "src/cuda/timex_cuda.cu"],
    verbose=False,
    extra_cuda_cflags=[
        "--use_fast_math",
        "--extra-device-vectorization",
        f"-DTmax={T_MAX}",
        f"-DBF={B_GROUP_FORWARD}",
        f"-DBB={B_GROUP_BACKWARD}",
    ],
)


class TimeX(PyLayer):
    @staticmethod
    def forward(ctx, w, k, B, C, T, eps):
        ctx.B = B
        ctx.C = C
        ctx.T = T
        assert (
            ctx.T % 4 == 0
            and ctx.T <= T_MAX
            and ctx.B % B_GROUP_FORWARD == 0
            and ctx.B % B_GROUP_BACKWARD == 0
        )
        ctx.save_for_backward(w, k)
        wk = timex_cuda.forward(w, k, B, C, T, eps)
        return wk

    @staticmethod
    def backward(ctx, gwk):
        assert (
            ctx.T % 4 == 0
            and ctx.T <= T_MAX
            and ctx.B % B_GROUP_FORWARD == 0
            and ctx.B % B_GROUP_BACKWARD == 0
        )
        w, k = ctx.saved_tensor()
        gw, gk = timex_cuda.backward(w, k, gwk, ctx.B, ctx.C, ctx.T)
        return gw.sum(axis=0), gk
