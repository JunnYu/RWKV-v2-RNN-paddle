########################################################################################################
# The RWKV v2-RNN Language Model - https://github.com/BlinkDL/RWKV-LM
########################################################################################################


import math
import logging
import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from paddle.nn.initializer import Assign #,Orthogonal
from paddle.optimizer import Adam
logger = logging.getLogger(__name__)



default_dtype = paddle.get_default_dtype()

use_fast = True

if use_fast:
    from src.timex import TimeX
else:
    class TimeX:
        @staticmethod
        def apply(w, k, B, C, T, eps):
            return F.conv1d(F.pad(k, pad=[T-1,0], data_format="NCL"), w.unsqueeze(1), groups=C, data_format="NCL") + eps

########################################################################################################
# RWKV: RWKV Time-mix + RWKV Channel-mix
########################################################################################################


RWKV_K_CLAMP = 60  # e^60 = 1e26
RWKV_K_EPS = 1e-16
RWKV_HEAD_QK_DIM = 256


def RWKV_Init(module, config):  # fancy initialization of all lin & emb layer in the module
    for m in module.children():
        if not isinstance(m, (nn.Linear, nn.Embedding)):
            continue
        with paddle.no_grad():
            name = '[unknown weight]'
            for name, parameter in module.named_parameters():  # find the name of the weight
                if id(m.weight) == id(parameter):
                    break

            shape = m.weight.shape
            gain = 1.0
            scale = 1.0  # extra scale for gain

            if isinstance(m, nn.Embedding):
                gain = math.sqrt(max(shape[0], shape[1]))
                if shape[0] == config.vocab_size and shape[1] == config.n_embd:  # token emb?
                    scale = 1e-4
                else:
                    scale = 0

            if isinstance(m, nn.Linear):
                if m.bias is not None:
                    m.bias.set_value(paddle.zeros_like(m.bias))
                if shape[0] > shape[1]:
                    gain = math.sqrt(shape[0] / shape[1])
                if shape[0] == config.vocab_size and shape[1] == config.n_embd:  # final projection?
                    scale = 0.5

            if hasattr(m, 'scale_init'):
                scale = m.scale_init

            # print(str(shape[0]).ljust(5), str(shape[1]).ljust(5), f'{round(scale,2):g}'.ljust(4), name)

            gain *= scale
            if scale == -999:
                m.weight.set_value(paddle.eye(m.weight.shape))
            elif gain == 0:
                # zero init is great for some RWKV matrices
                m.weight.set_value(paddle.zeros_like(m.weight))
            elif gain > 0:
                pass
                # TODO
                # orthogonal = Orthogonal(gain=gain)
                # orthogonal(m.weight)
            else:
                m.weight.set_value(paddle.normal(mean=0.0, std=-scale, shape=m.weight.shape))


class RWKV_TimeMix(nn.Layer):
    def __init__(self, config, layer_id):
        super().__init__()
        self.layer_id = layer_id
        self.ctx_len = config.ctx_len
        self.n_embd = config.n_embd

        attn_sz = config.n_embd

        ############# fancy init of time_w curves ###################################
        f1_begin = 3.0
        f1_end = 1.2
        f2_begin = 0.65
        f2_end = 0.4
        with paddle.no_grad():  # initial time_w curves for better convergence
            decay_speed = paddle.ones(shape=(attn_sz, 1))
            first_sa_layer_id = 1
            for h in range(attn_sz):
                f1 = f1_begin + (layer_id-first_sa_layer_id) / \
                    (config.n_layer-1-first_sa_layer_id) * (f1_end - f1_begin)
                f2 = f2_begin + (layer_id-first_sa_layer_id) / \
                    (config.n_layer-1-first_sa_layer_id) * (f2_end - f2_begin)
                if layer_id == first_sa_layer_id:
                    f1 += 0.5
                if layer_id == config.n_layer-2:
                    f2 = 0.4
                if layer_id == config.n_layer-1:
                    f2 = 0.37
                decay_speed[h][0] = math.pow(f2, h / (attn_sz-1) * 7) * f1
        self.time_decay = self.create_parameter(decay_speed.shape, dtype=default_dtype, default_initializer=Assign(paddle.log(decay_speed))) # will use exp(self.time_decay) to ensure time_decay > 0
        self.time_curve = paddle.to_tensor(
            [-(config.ctx_len - 2 - i) for i in range(config.ctx_len-1)]).unsqueeze(0)

        self.time_first = self.create_parameter(shape=(attn_sz, 1),dtype=default_dtype, default_initializer=Assign(paddle.ones((attn_sz, 1)) * math.log(0.3)))
        #############################################################################

        self.time_shift = nn.Pad2D([1, 0], data_format="NLC")
        with paddle.no_grad():  # init to "shift half of the channels"
            ww = paddle.ones((1, 1, config.n_embd))
            for i in range(config.n_embd // 2):
                ww[0, 0, i] = 0
        self.time_mix = self.create_parameter(shape=ww.shape, dtype=default_dtype, default_initializer=Assign(ww))  

        self.key = nn.Linear(config.n_embd, attn_sz, bias_attr=False)
        self.value = nn.Linear(config.n_embd, attn_sz, bias_attr=False)
        self.receptance = nn.Linear(config.n_embd, attn_sz, bias_attr=False)

        self.output = nn.Linear(attn_sz, config.n_embd, bias_attr=False)

        self.key.scale_init = 0
        self.receptance.scale_init = 0
        self.output.scale_init = 0



    def forward(self, x):
        B, T, C = x.shape

        x = x * self.time_mix + self.time_shift(x[:,:-1]) * (1 - self.time_mix)

        k = self.key(x).transpose([0,2,1])
        v = self.value(x).transpose([0,2,1])
        r = self.receptance(x)

        # RWKV_K_CLAMP can be removed if the CUDA kernel substracts the correct k_max for each k (I will do this later)
        k = paddle.clip(k, max=RWKV_K_CLAMP)
        k = paddle.exp(k)
        kv = k * v

        self.time_w = paddle.concat(
            [paddle.exp(self.time_decay) * self.time_curve, self.time_first], axis=-1)
        w = paddle.exp(self.time_w)

        ####################TimeX
        wkv = TimeX.apply(w, kv, B, C, T, 0)
        # RWKV_K_EPS can be removed if the CUDA kernel sets 0/0 = 0 (I will do this later)
        wk = TimeX.apply(w, k, B, C, T, RWKV_K_EPS)

        rwkv = F.sigmoid(r) * (wkv / wk).transpose([0,2,1])
        rwkv = self.output(rwkv)
        return rwkv


class RWKV_ChannelMix(nn.Layer):
    def __init__(self, config, layer_id):
        super().__init__()
        self.layer_id = layer_id

        self.time_shift = nn.Pad2D([1, 0], data_format="NLC")

        with paddle.no_grad():  # init to "shift half of the channels"
            x = paddle.ones((1, 1, config.n_embd))
            for i in range(config.n_embd // 2):
                x[0, 0, i] = 0
        self.time_mix = self.create_parameter(shape=x.shape, dtype=default_dtype, default_initializer=Assign(x)) 

        hidden_sz = 4 * config.n_embd
        self.key = nn.Linear(config.n_embd, hidden_sz, bias_attr=False)
        self.receptance = nn.Linear(config.n_embd, config.n_embd, bias_attr=False)
        self.value = nn.Linear(hidden_sz, config.n_embd, bias_attr=False)

        self.value.scale_init = 0
        self.receptance.scale_init = 0

    def forward(self, x):
        x = x * self.time_mix + self.time_shift(x[:, :-1]) * (1 - self.time_mix)

        k = self.key(x)
        k = paddle.square(F.relu(k))
        kv = self.value(k)

        rkv = F.sigmoid(self.receptance(x)) * kv
        return rkv

########################################################################################################
# The GPT Model with our blocks
########################################################################################################


class GPTConfig:
    def __init__(self, vocab_size, ctx_len, **kwargs):
        self.vocab_size = vocab_size
        self.ctx_len = ctx_len
        for k, v in kwargs.items():
            setattr(self, k, v)


class Block(nn.Layer):
    def __init__(self, config, layer_id):
        super().__init__()
        self.config = config
        self.layer_id = layer_id

        self.ln1 = nn.LayerNorm(config.n_embd)
        self.ln2 = nn.LayerNorm(config.n_embd)

        if self.layer_id == 0 and self.config.model_type == 'RWKV-ffnPre':
            self.ffnPre = RWKV_ChannelMix(config, layer_id+1000)
        else:
            self.att = RWKV_TimeMix(config, layer_id)

        self.ffn = RWKV_ChannelMix(config, layer_id)

    def forward(self, x):
        x = self.ln1(x)
        if self.layer_id == 0 and self.config.model_type == 'RWKV-ffnPre':
            x = x + self.ffnPre(x)  # better in some cases
        else:
            x = x + self.att(x)
        x = self.ln2(x)
        x = x + self.ffn(x)
        return x


class GPT(nn.Layer):
    def __init__(self, config):
        super().__init__()
        self.step = 0
        self.config = config

        self.emb = nn.Embedding(config.vocab_size, config.n_embd)

        self.blocks = nn.Sequential(*[Block(config, i)
                                    for i in range(config.n_layer)])

        self.ln_out = nn.LayerNorm(config.n_embd)
        self.head = nn.Linear(config.n_embd, config.vocab_size, bias_attr=False)

        self.head_q = nn.Linear(config.n_embd, RWKV_HEAD_QK_DIM, bias_attr=False)
        self.head_q.scale_init = 0
        self.head_k = nn.Linear(config.n_embd, RWKV_HEAD_QK_DIM, bias_attr=False)
        self.head_k.scale_init = 0.1
        self.register_buffer("copy_mask", paddle.tril(
            paddle.ones((config.ctx_len, config.ctx_len))))

        self.ctx_len = config.ctx_len

        RWKV_Init(self, config)

        logger.info("number of parameters: %e", sum(p.numel()
                    for p in self.parameters()))

    def get_ctx_len(self):
        return self.ctx_len

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear)):
            module.weight.set_value(paddle.normal(mean=0.0, std=0.01, shape=module.weight.shape))
        if isinstance(module, (nn.Embedding)):
            module.weight.set_value(paddle.normal(mean=0.0, std=1e-5, shape=module.weight.shape))
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.set_value(paddle.zeros_like(module.bias))


    def configure_optimizers(self, train_config):
        # separate out all parameters to those that will and won't experience regularizing weight decay
        decay = set()
        no_decay = set()

        for mn, m in self.named_children():  # here we disable weight_decay
            for pn, p in m.named_parameters():
                fpn = '%s.%s' % (mn, pn) if mn else pn  # full param name
                no_decay.add(fpn)

        param_dict = {pn: p for pn, p in self.named_parameters()}
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert len(
            inter_params) == 0, "parameters %s made it into both decay/no_decay sets!" % (str(inter_params), )
        assert len(param_dict.keys() - union_params) == 0, "parameters %s were not separated into either decay/no_decay set!" \
            % (str(param_dict.keys() - union_params), )

        optim_groups = [
            {"params": [param_dict[pn]
                        for pn in sorted(list(no_decay))], "weight_decay": 0.0},
        ]
        if train_config.grad_norm_clip > 0:
            grad_clip = nn.ClipGradByGlobalNorm(clip_norm=train_config.grad_norm_clip)

        optimizer = Adam(
            parameters=optim_groups, learning_rate=train_config.learning_rate, beta1=train_config.betas[0], beta2=train_config.betas[1], epsilon=train_config.eps, grad_clip=grad_clip)

        return optimizer

    def forward(self, idx, targets=None):
        self.step += 1
        B, T = idx.shape
        assert T <= self.ctx_len, "Cannot forward, because len(input) > model ctx_len."
        x = self.emb(idx)

        x = self.blocks(x)

        x = self.ln_out(x)

        q = self.head_q(x)[:, :T, :]
        k = self.head_k(x)[:, :T, :]
        c = (q @ k.transpose([0, 2, 1])) * (1.0 / RWKV_HEAD_QK_DIM)
        c = paddle.where(self.copy_mask[:T, :T] == 0, paddle.zeros((1,)), c)
  
        c = c @ F.one_hot(idx, num_classes=self.config.vocab_size).astype(x.dtype)
        x = self.head(x) + c

        loss = None
        if targets is not None:
            loss = F.cross_entropy(x.reshape((-1, x.shape[-1])), targets.reshape((-1,)))

        return x, loss


if __name__ == "__main__":
    paddle.set_device("gpu")
    config = GPTConfig(vocab_size=6064, ctx_len=1024, model_type="RWKV", n_layer=6, n_embd=512)
    model = GPT(config)
    # model.load_dict(paddle.load("weights/enwik8-ppl1.65-6064-1024-RWKV-6-512-2022-03-25-21-05-13.pdparams"))
    pdx = paddle.arange(1024*4).reshape((4,1024)).astype("int64")
    with paddle.no_grad():
        pdout = model(pdx)[0]
    print(pdout.shape)
    

    
