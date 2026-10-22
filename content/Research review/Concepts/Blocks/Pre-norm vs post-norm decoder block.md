---
title: Pre-norm vs post-norm decoder block
draft: false
paper: https://arxiv.org/pdf/2002.04745
date: 2026-10-21
---
## Definition

A transformer block is made of two **sublayers**, a multi-head attention and a feed-forward network (FFN, also called MLP), each wrapped by a **residual connection** (the "Add") and a [[Layer Normalization]] (the "Norm"). The two variants only differ by **where the normalization sits**:

- **Post-norm** (original Transformer, "Attention Is All You Need"; also BERT): the Norm is applied *after* the Add, so it sits **on the main path**. The sum of the input and of the sublayer output is renormalized.
- **Pre-norm** (GPT-2, Llama, Qwen, Mistral...): the Norm is applied *before* the sublayer, on the **side branch only**. The main path is a plain sum with no normalization in the middle of the network, except for one final Norm before the output (unembedding) layer.

In a pre-norm model, the main path is what is called the **residual stream**: every sublayer reads a normalized copy of it and adds its output back.

## Formula

Let $x$ be the input of the block, $\text{Attn}$ the attention sublayer, $\text{FFN}$ the feed-forward sublayer and $\text{LN}$ the Layer Normalization.

**Post-norm**:
$$
\begin{align}
h &= \text{LN}\big(x + \text{Attn}(x)\big)\notag\\
y &= \text{LN}\big(h + \text{FFN}(h)\big)\notag
\end{align}
$$

**Pre-norm**:
$$
\begin{align}
h &= x + \text{Attn}\big(\text{LN}(x)\big)\notag\\
y &= h + \text{FFN}\big(\text{LN}(h)\big)\notag
\end{align}
$$

let's break the difference down:
- in **post-norm**, $\text{LN}$ is outside of the parentheses of the residual sum: the output of every block is renormalized, so the identity path $x \to y$ is repeatedly rescaled and recentered.
- in **pre-norm**, $\text{LN}$ is inside the sublayer branch: the identity path stays untouched, so after $L$ blocks the output is $x_0 + \sum_{\ell}(\text{sublayer outputs})$, a sum of contributions living in the same space.

## Usage

- **Training stability**: [Xiong et al. (2020)](https://arxiv.org/abs/2002.04745) show that post-norm has large gradients near the output layer at initialization, which makes a learning-rate warm-up mandatory, while pre-norm has well-behaved gradients and can be trained deeper and with less (or no) warm-up. This is why almost all modern LLMs are pre-norm.
- **Trade-off**: pre-norm is easier to train but the norm of the residual stream tends to grow with depth, so the contribution of later layers is relatively smaller. Post-norm is harder to train but is often reported to give slightly better final quality when it does converge.
- **Variants**: modern models often replace LayerNorm by RMSNorm (no mean subtraction). [Gemma 2](https://arxiv.org/pdf/2408.00118) and 3 use a *sandwich* version, with a Norm before **and** after each sublayer; the second Norm is applied to the sublayer output *before* it is added, so the residual stream itself still has no Norm on its path.
- **Interpretability**: the pre-norm layout is what makes the "residual stream" view meaningful: the stream is a running sum, so a direction can be read (projection) or written (adding a vector, as in activation steering) at any layer without being rescaled by a following Norm. See [[The pain axis - LLMs represent self-directed harm and act on it]] for an example where a concept vector is added to the stream of 25 pre-norm models.
