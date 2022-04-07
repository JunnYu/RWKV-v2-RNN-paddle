from paddle.utils.cpp_extension import CUDAExtension, setup

setup(
    name="paddle_custom_ops",
    ext_modules=CUDAExtension(
        sources=["./timex_op.cc", "./timex_cuda.cu"],
        verbose=True,
        extra_cuda_cflags=[
            "--use_fast_math",
            "--extra-device-vectorization",
        ],
    ),
)
