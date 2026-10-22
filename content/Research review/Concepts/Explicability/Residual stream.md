---
title: Residual stream
draft: false
paper: https://transformer-circuits.pub/2021/framework/index.html
date: 2026-10-21
---
## Definition

The **residual stream** is the running vector that a transformer carries from its input to its output. It starts as the embedding of the token, then every attention and MLP sublayer **reads** a (normalized) copy of it and **adds** its output back. Nothing is ever overwritten: the stream at layer $\ell$ is the token embedding plus the sum of everything written by the layers below.

Three things to keep in mind:

- **One stream per token position.** For a sequence of $T$ tokens, the activations at layer $\ell$ have shape `(T, d_model)`. Each token has its own stream vector, and attention is the only operation that lets a position read the streams of other positions.
- **It lives in $\mathbb{R}^{d_{model}}$**, a few thousand dimensions (e.g. 4,096 for Llama 3.1 8B), not in the space of the billions of parameters. Its coordinate axes have no particular meaning: concepts are encoded as *directions* spread over many coordinates.
- **It requires a pre-norm architecture to be a clean running sum**: the Norm sits on the side branch, not on the main path (see [[Pre-norm vs post-norm decoder block]]).

![[residual_stream_0.svg]]

## Formula

Let $h_{\ell-1}$ be the stream entering layer $\ell$, $\text{LN}$ a normalization, $\text{Attn}$ the attention sublayer and $\text{MLP}$ the feed-forward sublayer:

$$
\begin{align}
h_0 &= \text{embed}(x_t)\notag\\
\vdots &\notag\\
h_\ell^{mid} &= h_{\ell-1} + \text{Attn}_\ell\big(\text{LN}(h_{\ell-1})\big)\notag\\
h_\ell &= h_\ell^{mid} + \text{MLP}_\ell\big(\text{LN}(h_\ell^{mid})\big)\notag\\
\vdots &\notag\\
\text{logits} &= W_U\,\text{LN}_f(h_L)\notag
\end{align}
$$

Unrolling the recursion, with $a_\ell$ and $m_\ell$ the outputs of the attention and MLP of layer $\ell$:

$$
h_L = h_0 + \sum_{\ell=1}^{L}\big(a_\ell + m_\ell\big)
$$

let's break this formula step-by-step:
- $h_0$ is the embedding of the current token (the starting point of the sum).
- $a_\ell$ and $m_\ell$ are the contributions written by layer $\ell$. The final stream is the sum of all of them, so each layer's contribution can be read separately.
- $\text{LN}_f$ is the final normalization applied once before the unembedding matrix $W_U$ (shape `(|V|, d_model)`, see [[Unembedding layer]]), which turns the stream into one score (logit) per vocabulary token.

What each component does to the stream:

- **LayerNorm** ([[Layer Normalization]]) works on a **copy** of the stream before each sublayer. It subtracts the mean over the features and rescales to a fixed norm ($\approx\sqrt{d_{model}}$), then applies a learned scale $\gamma$ and shift $\beta$. The rescaling itself keeps the direction and discards the magnitude, but the mean removal and the element-wise $\gamma$ and $\beta$ rotate the vector slightly. RMSNorm, used by Llama, Qwen, Mistral and Gemma, has no mean removal ($x/\text{RMS}(x)\cdot\gamma$). The stream itself keeps its norm.
- **Attention** computes, for each head, a weighted average of vectors read from the streams of the current and earlier tokens: $o_t = \sum_{s\leq t}\alpha_{ts}\,W_V\,\text{LN}(h_s)$, with weights $\alpha_{ts}=\text{softmax}_s(q_t\cdot k_s/\sqrt{d_k})$. The heads' outputs are projected back to the stream by $W_O$ and added. It is the only step where information moves **between positions**.
- **MLP** acts on **one position only**: $m = W_{out}\,\phi(W_{in}\,\text{LN}(h))$ (*note: modern models use a gated version*). Each hidden neuron *reads* the stream along one direction (a row of $W_{in}$), fires depending on how aligned the stream is with it, and *writes* a fixed direction (a column of $W_{out}$) into the stream, scaled by its activation. It is non-linear, but its output is still just added.

## Example

The figure below is a toy 2D version ($d_{model}=2$, real models have thousands of dimensions, and the numbers are illustrative). It follows **one stream $h$** through the four steps: the same $h$ is normalized, read by attention, then updated, and the result is read by the MLP.

![[residual_stream_1.svg]]

## Usage

- **Reading a concept**: since the stream is a sum in a single space, the projection $h_\ell\cdot\hat{v}$ at any layer measures how much of the concept direction $\hat{v}$ is present. This is what difference-in-means probes do.
- **Writing a concept (steering)**: adding $\alpha\hat{v}$ to $h_\ell$ at one layer pushes the model along that direction, and the following layers read it as if a block had written it. Because the stream is never renormalized on its main path, the added vector is not rescaled in the stream itself. The sublayers still read a normalized copy, so its influence depends on its size relative to the stream norm (next point).
- **Norm matters**: in a pre-norm model the stream norm tends to grow with depth, so the same added vector weighs less in late layers.
- **Reading the output**: the final stream goes through $W_U$ to give logits, so applying $W_U$ to a direction shows which tokens it promotes or suppresses (the *logit lens* idea).
- **Analyzing circuits**: because the output is a sum of per-layer contributions, one can measure which attention heads or MLPs write the most along a given direction, which is the basis of the transformer-circuits line of work.
