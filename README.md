# RWKV-v2-RNN-paddle
 paddle version RWKV-v2-RNN

# TODO
- [x] 完善paddle版本训练代码（模型已经搭建了）。
- [x] 封装paddle自定义算子，实现depthwise convolution 1d。

# 训练
- 下载enwik8.zip, `wget https://data.deepai.org/enwik8.zip` 
- 解压enwik8.zip, `unzip enwik8.zip`
- 训练，`python train.py`
- 如果编译自定义算子报错的话，"--use_fast_math", "--extra-device-vectorization" 注释掉。

# 测试
- 安装paddle
- 下载权重
- 转换权重，`python convert.py`。
- 测试生成效果，`python run.py`

```bash
Loading weights/enwik8-ppl1.65-6064-1024-RWKV-6-512-2022-03-25-21-05-13...

Your prompt has 7 tokens.

--> Currently the first run takes a while if your prompt is long, as we are using RNN to process the prompt. This will be much faster in future versions. <--

------------------------------
In the [[1960s]], the name &quot;Dessert&quot; was coined to designate the designers &quot;Dessert&quot; and &quot;The Dessert&quot; by the [[West and West Indies|West Australian]] sentiment inside the [[Balkans]]. In [[1965]] [[George Colley]] and [[John Douglas (James Maas)|John Colleen]] were drawn from the working class to the [[United States]] and in [[1966]] the first [[classified information|classified]] products of [[subsistence farming]]. In [[1968]], [[Robert A. Heinlein]] published ''[[Natu
---------- 19.66s ------------------------------
In the past, '''enthusiastic community''' often referred to as '''enthusiastic''' or '''anthroposophical communication''' and derives its nature from the influence of [[Aristotle]], the [[early epoch]] of which is still present in the early complex phase of ethical doctrine.

An enthusiasm for epihelia-night alternative cultures may remain popular in school until early ???? --&gt;

== Etymology ==
The word &quot;ethheim&quot; is derived from the [[Latin language|Latin]] ''entheosis'', meaning &quot;wi
---------- 19.91s ------------------------------
In the [[1980s]], the term &quot;hollow earth&quot; has inspired several somewhat less well-known societies.  In the post-war sense of the times, the study of hollow-earth habits in the west has also developed in an inconsistent state of elementary mathematics.  The most influential best example is the terminology of the [[projective special calculation]] of [[antiderivative|spring]]s and [[antiderivative]]s.

== Consequences ==

=== Electronic media ===
{{main|open question article}}
The [[constant f
---------- 19.46s ------------------------------
In the late 1980s, a new studio audience came onto &quot;[[Sound safest9eitssons|sound tracks]]&quot;, which were re-issued with the album ''[[Down in the Year (song)|Down the Avenger]]''.  The theme for the film also received a more [[piano music|piano sonata]] from the classic novel ''[[Stars and Stripes (novel)|Star Trek]]'', which was also the most popular song in the US.  

Since the early 1980s, the regional records became quite popular and influential. In the late 1970s and 1980s a great deal h
---------- 19.4s ------------------------------
In the past, curiosities were created by following parties in [[North America]] and [[South America]], such as [[South Carolina]] and [[Mexico]]. Some people have suggested that these mere measures of status are not satisfied as the measures are: the [[European Convention on Human Rights]] (FCU) and the [[European Investment Fund]] (EEU) and the [[International Ethnic Policy Center]] (IFCL). The European Commission also advocates a comprehensive development and self-reflective effort to impose a relat
---------- 19.54s ------------------------------
In the 2005 [[London 2012 Olympics]], Beckham was one of the few weekly past ten years earlier to become the [[popular beta]] for working the club. Because of this, Beckham has published many of his selling competitions on play worldwide. In 1998, Beckham received a [[Professional football (soccer) play|Professional Player of the Professional Professional]] to receive the [[Great White Show]] on his went for 12 years.

==External links==
*[http://www.roguesting.ca/ Rogerian Institute] - Official site

---------- 19.2s ------------------------------
```

------------------------------------------------------------------------
# 原作者readme: he RWKV Language Model

## RWKV v2: RNN with Transformer Performance

RWKV v2 is a RNN which can also be directly trained like a GPT transformer. You only need x_t, a_t, b_t of position t to compute the vectors for position t+1. Hence it can be 100x faster than GPT, and 100x more VRAM friendly.

See the release for a **27M params model on enwik8 with 0.72 BPC(dev)**.

![RWKV-v2-RNN](RWKV-v2-RNN-run.png)

## How it works

The a b c d factors work together to build a time-decay curve: X, 1, W, W^2, W^3, ...

Write out the formulas for "token at pos 2" and "token at pos 3" and you will get the idea:
* a and b: EMAs of kv and k.
* c and d: a and b combined with self-attention.

kv / k is the memory mechanism. The token with high k can be remembered for a long duration, if W is close to 1 in the channel.

It's also using my SmallInitEmb trick https://github.com/BlinkDL/SmallInitEmb (applicable to all transformers), and a custom CUDA kernel https://github.com/BlinkDL/RWKV-CUDA .

I find it might be nice to make the model stay on a mid-lr for a long period, because in theory that's where most learning shall happen. For example: 6e-4 to 1e-4 in 15% of steps, stays on 1e-4 for 60% of steps (actually I monitor the loss and decay the lr when it plateaus), then 1e-4 to 1e-5 in 25% of steps.

The pseudocode (execution from top to bottom):

![RWKV-v2-RNN](RWKV-v2-RNN.png)

## v1

We propose the RWKV language model, with alternating time-mix and channel-mix layers:

<img src=
"https://render.githubusercontent.com/render/math?math=%5Cdisplaystyle+%5Cbegin%7Balign%2A%7D%0A%5Ctext%7BTime-mix+%3A%7D+%26%26+%5Ctext%7BTM%7D_%7Bt%2Cc%7D+%26%26%3D%26%26%5Ctext%7Bsigmoid%7D%28%5Ctext%7BR%7D_%7Bt%2Cc%7D%29+%26%26%5Ccdot%26%26+%26%26%5Ctextstyle%5Csum_%7Bu%7D+%26%26%5Ctextbf%7BW%7D_%7Bt%2Cu%2Cc%7D+%26%26%5Ccdot%26%26+%5Ctext%7Bsoftmax%7D_t%28%5Ctext%7BK%7D_%7Bu%2Cc%7D%29+%26%26%5Ccdot%26%26+%5Ctext%7BV%7D_%7Bu%2Cc%7D%5C%5C%0A%5Ctext%7BChannel-mix+%3A%7D+%26%26+%5Ctext%7BCM%7D_%7Bt%2Cc%7D+%26%26%3D%26%26%5Ctext%7Bsigmoid%7D%28%5Ctext%7BR%7D_%7Bt%2Cc%7D%29+%26%26%5Ccdot%26%26+%26%26%5Ctextstyle%5Csum_d+%26%26%5Ctextbf%7BW%7D_%7Bc%2Cd%7D+%26%26%5Ccdot%26%26+%5Ctext%7Bgelu%7D%28%5Ctext%7BK%7D_%7Bt%2Cd%7D%29+%26%26%5Ccdot%26%26+%5Ctext%7BV%7D_%7Bt%2Cd%7D%0A%5Cend%7Balign%2A%7D%0A" 
alt="\begin{align*}
\text{Time-mix :} && \text{TM}_{t,c} &&=&&\text{sigmoid}(\text{R}_{t,c}) &&\cdot&& &&\textstyle\sum_{u} &&\textbf{W}_{t,u,c} &&\cdot&& \text{softmax}_t(\text{K}_{u,c}) &&\cdot&& \text{V}_{u,c}\\
\text{Channel-mix :} && \text{CM}_{t,c} &&=&&\text{sigmoid}(\text{R}_{t,c}) &&\cdot&& &&\textstyle\sum_d &&\textbf{W}_{c,d} &&\cdot&& \text{gelu}(\text{K}_{t,d}) &&\cdot&& \text{V}_{t,d}
\end{align*}
">

* The R, K, V are generated by linear transforms of input, and W is parameter. The idea of RWKV is to decompose attention into R(target) * W(src, target) * K(src). So we can call R "receptance", and sigmoid means it's in 0~1 range.

* The Time-mix is similar to AFT (https://arxiv.org/abs/2105.14103). There are two differences.

(1) We changed the normalization (denominator). For masked language models, we define:

<img src=
"https://render.githubusercontent.com/render/math?math=%5Cdisplaystyle+%5Ctext%7Bsoftmax%7D_t%28%5Ctext%7BK%7D_%7Bu%2Cc%7D%29+%3D+%5Cfrac%7B%5Cexp%28%5Ctext%7BK%7D_%7Bu%2Cc%7D%29%7D%7B%5Csum_%7Bv+%5Cleq+t%7D%5Cexp%28%5Ctext%7BK%7D_%7Bv%2Cc%7D%29%7D" 
alt="\text{softmax}_t(\text{K}_{u,c}) = \frac{\exp(\text{K}_{u,c})}{\sum_{v \leq t}\exp(\text{K}_{v,c})}">

**(UPDATE: We are using the original AFT normalization in v2)**
 
Initialize K and R matrices (and the output projection matrix) to ZERO for fast & stable convergence.
 
(2) We decompose W_{t,u,c} and introduce multi-head W (here h is the corresponding head of c):

<img src=
"https://render.githubusercontent.com/render/math?math=%5Cdisplaystyle+W_%7Bt%2Cu%2Cc%7D%3Df_h%28t-u%29%5Ccdot+%5Calpha_h%28u%29+%5Ccdot+%5Cbeta_h%28t%29" 
alt="W_{t,u,c}=f_h(t-u)\cdot \alpha_h(u) \cdot \beta_h(t)">

Moreover we multiply the final output of Time-mix layer by γ(t). The reason for the α β γ factors, is because the context size is smaller when t is small, and this can be compensated using the α β γ factors.

**(UPDATE: We remove α β γ factors in v2-RNN and restrict W to be of a simple form and hence able to rewrite it as RNN)**

* The Channel-mix is similar to GeGLU (https://arxiv.org/abs/2002.05202) with an extra R factor. Initialize R and W matrices to ZERO for fast & stable convergence.

* Finally, we add extra token-shift (time-shift mixing) as in (https://github.com/BlinkDL/minGPT-tuned).

# Token-shift (time-shift mixing)

The token-shift explicitly uses (half the channels of this token) & (half the channels of prev token) to generate all vectors (QKV, RWKV, ...).

```
self.time_shift = nn.ZeroPad2d((0,0,1,-1))

x = torch.cat([self.time_shift(x[:, :, :C//2]), x[:, :, C//2:]], dim = -1)
```

Dividing channels by 2 and shift-1 works great for char-level English and char-level Chinese LM.

However for BPE-level English LM, it's only effective if your embedding is large enough (at least 1024 - so the usual small L12-D768 model is not enough).

My theory on the effectiveness of token-shift:

When we train a GPT, the hidden representation of a token has to accomplish two different objects:

1. Predict the next token. Sometimes this is easy (obvious next token).

2. Collect all previous context info, so later tokens can use it. This is always hard.

The shifted channels can focus on (2), so we have good propagation of info. It's like some kind of residual connection, or a small RNN inside the transformer.

You can use token-shift in usual QKV self-attention too. I looked at the weights, and found V really likes the shifted channels, less so for Q. Makes sense if you think about it. I also found you may want to use less mixing in higher layers.

p.s. There is a MHA_pro model in this repo with strong performance. Give it a try :)

# The Head-QK Trick: learning to copy and avoid tokens

In usual transformer, a small model has difficulty copying tokens (such as person names) in the context. We add extra Q & K to the final output such that the model can directly copy (or avoid) tokens in the context. Afterwards the model will teach itself NER (named entity recognition) if you look at the learned weights.
```
q = self.head_q(x)[:,:T,:]
k = self.head_k(x)[:,:T,:]
c = (q @ k.transpose(-2, -1)) * (1.0 / 256)
c = c.masked_fill(self.copy_mask[:T,:T] == 0, 0)
c = c @ F.one_hot(idx, num_classes = self.config.vocab_size).float()       
x = self.head(x) + c
```

# The top-a Sampling method

We also propose a new sampling method called top-a (as in src/utils.py):

(1) Find the max probability p_max after softmax.

(2) Remove all entries whose probability is lower than 0.02 * pow(p_max, 2). So it's adaptive, hence "top-a".

(3) Feel free to tune the 0.02 and 2 factor. Tune 0.02 first.

The idea of top-a:
1. If max_prob=0.9, then remove all tokens with prob < 0.0162 (so, removing most alternatives)
2. If max_prob=0.5, then remove all tokens with prob < 0.0050 (so, allowing more choices)
3. If max_prob=0.1, then remove all tokens with prob < 0.0002 (so, allowing lots of possibilities)

```
probs = F.softmax(logits, dim=-1)

limit = torch.pow(torch.max(probs), 2) * 0.02
logits[probs < limit] = -float('Inf')
```

# Performance

Character-level loss on simplebooks-92 dataset https://dldata-public.s3.us-east-2.amazonaws.com/simplebooks.zip

![RWKV-vs-MHA](RWKV-vs-MHA.png)

Gray: usual MHA+Rotary+GeGLU - performance not as good. 17.2M params.

Red: RWKV ("linear" attention) - VRAM friendly - quite faster when ctx window is long - good performance. 16.6M params.

Green: MHA+Rotary+GeGLU+Token_shift. 17.2M params.

Blue: MHA_pro (MHA with various tweaks & RWKV-type-FFN) - slow - needs more VRAM - good performance. 16.6M params.

```
@software{peng_bo_2021_5196578,
  author       = {PENG Bo},
  title        = {BlinkDL/RWKV-LM: 0.01},
  month        = aug,
  year         = 2021,
  publisher    = {Zenodo},
  version      = {0.01},
  doi          = {10.5281/zenodo.5196577},
  url          = {https://doi.org/10.5281/zenodo.5196577}
}
```

# Initialization

We use careful initialization for RWKV to get fast convergence - orthogonal matrices with proper scaling, and special time_w curves. Check model.py for details.

Some learned time_w examples:

![RWKV-time-w](RWKV-time-w.png)
