from paddle.autograd import PyLayer
from paddle_custom_ops import TimeX_forward, TimeX_backward

T_MAX = 1024  # increase this if your ctx_len > 1024
B_GROUP_FORWARD = 4  # set to 8 for best performance
B_GROUP_BACKWARD = 2  # set to 2 for best performance

# 通过创建`PyLayer`子类的方式实现动态图Python Op
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
        wk = TimeX_forward(w, k, B, C, T, eps)
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
        # 调用Paddle API自定义反向计算
        gw, gk = TimeX_backward(w, k, gwk, ctx.B, ctx.C, ctx.T)
        # forward只有一个Tensor输入，因此，backward只有一个输出。
        return gw.sum(axis=0), gk
