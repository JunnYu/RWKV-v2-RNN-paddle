import types
import copy
import paddle
import paddle.nn.functional as F

RWKV_K_CLAMP = 60
RWKV_K_EPS = 1e-16
RWKV_HEAD_QK_DIM = 256

DEBUG_TIME = False   # True False - show trained time-coeffs


class RWKV_RNN():
    def __init__(self, MODEL_NAME, RUN_DEVICE, model_type, n_layer, n_embd, ctx_len):
        self.RUN_DEVICE = RUN_DEVICE
        self.model_type = model_type
        self.n_layer = n_layer
        self.n_embd = n_embd
        self.ctx_len = ctx_len

        self.w = types.SimpleNamespace()

        w = paddle.load(MODEL_NAME + '.pdparams')
        for x in w.keys():
            if '.time_' in x:
                w[x] = w[x].squeeze()
            if '.time_decay' in x:
                w[x] = paddle.exp(-paddle.exp(w[x]))
            if '.time_first' in x:
                w[x] = paddle.exp(w[x])
            if DEBUG_TIME and '.time_' in x:
                print(x, w[x].squeeze().cpu().numpy())

            xx = x.split('.')
            here = self.w
            for i in range(len(xx)):
                if xx[i].isdigit():
                    ii = int(xx[i])
                    if ii not in here:
                        here[ii] = types.SimpleNamespace()
                    here = here[ii]
                else:
                    if i == len(xx) - 1:
                        setattr(here, xx[i], w[x])
                    elif not hasattr(here, xx[i]):
                        if xx[i+1].isdigit():
                            setattr(here, xx[i], {})
                        else:
                            setattr(here, xx[i], types.SimpleNamespace())
                    here = getattr(here, xx[i])

        self.clear()

    def clear(self):
        self.xx = {}
        self.aa = {}
        self.bb = {}
        self.hk = None

    def save(self, target):
        target.xx = copy.deepcopy(self.xx)
        target.aa = copy.deepcopy(self.aa)
        target.bb = copy.deepcopy(self.bb)
        target.hk = copy.deepcopy(self.hk)

    def load(self, target):
        self.xx = copy.deepcopy(target.xx)
        self.aa = copy.deepcopy(target.aa)
        self.bb = copy.deepcopy(target.bb)
        self.hk = copy.deepcopy(target.hk)

    def LN(self, xx, w):
        if xx.ndim == 1:
            # F.layer norm必须要有个batch size维度。
            return F.layer_norm(xx.unsqueeze(0), (self.n_embd,), weight=w.weight, bias=w.bias).squeeze(0)
        else:
            return F.layer_norm(xx, (self.n_embd,), weight=w.weight, bias=w.bias)

    def FF(self, xx, w, name):
        if name not in self.xx:
            self.xx[name] = paddle.zeros((self.n_embd,))
        x = xx * w.time_mix + self.xx[name] * (1 - w.time_mix)
        self.xx[name] = xx

        r = F.sigmoid(x @ w.receptance.weight)
        k = paddle.square(F.relu(x @ w.key.weight))
        kv = k @ w.value.weight

        return r * kv

    def SA(self, xx, w, name):
        if name not in self.xx:
            self.xx[name] = paddle.zeros((self.n_embd,))
            self.aa[name] = paddle.zeros((self.n_embd,))
            self.bb[name] = paddle.zeros((self.n_embd,))
        x = xx * w.time_mix + self.xx[name] * (1 - w.time_mix)
        self.xx[name] = xx
 
        r = F.sigmoid(x @ w.receptance.weight)

        k = paddle.exp(paddle.clip(x @ w.key.weight, max=RWKV_K_CLAMP))
        v = x @ w.value.weight
        kv = k * v

        a = self.aa[name] + w.time_first * kv
        b = self.bb[name] + w.time_first * k
        self.aa[name] = w.time_decay * self.aa[name] + kv
        self.bb[name] = w.time_decay * self.bb[name] + k

        rwkv = r * a / (b + RWKV_K_EPS)

        return rwkv @ w.output.weight

    def run(self, ctx):
        w = self.w
        x = w.emb.weight[ctx[-1]]

        for i in range(self.n_layer):
            x = self.LN(x, w.blocks[i].ln1)
            
            if i == 0 and self.model_type == 'RWKV-ffnPre':
                x = x + self.FF(x, w.blocks[i].ffnPre, f'ffnPre.{i}')
            else:
                x = x + self.SA(x, w.blocks[i].att, f'att.{i}')
            x = self.LN(x, w.blocks[i].ln2)
            x = x + self.FF(x, w.blocks[i].ffn, f'ffn.{i}')

        x = self.LN(x, w.ln_out)
        if self.hk is None:
            self.hk = (x @ w.head_k.weight).unsqueeze(0)
        else:
            self.hk = paddle.concat(
                [self.hk, (x @ w.head_k.weight).unsqueeze(0)], axis=0)
        if self.hk.shape[0] > self.ctx_len:
            self.hk = self.hk[-self.ctx_len:, :]

        q = x @ w.head_q.weight

        x = x @ w.head.weight
        x = x.cpu().numpy()

        c = (self.hk @ q) / RWKV_HEAD_QK_DIM
        for i in range(len(c)):
            x[ctx[i]] += c[i]

        return x
