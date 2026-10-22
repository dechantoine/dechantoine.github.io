---
title: Unembedding layer
draft: false
paper: 
date: 2026-10-22
---
## Definition

The **unembedding layer** is the last step of a language model: a single matrix $W_U$ of shape `(|V|, d_model)` that turns the final [[Residual stream]] into one score (a **logit**) per token of the vocabulary. It is the mirror of the embedding layer $W_E$, which maps a token id to a vector at the start of the model. Some models tie them (the same matrix serves as embedding and unembedding, e.g. Gemma), others learn them separately (e.g. Llama 3.1).

## Formula

Let $h_L$ be the residual stream of the last position after the last block, $\text{LN}_f$ the final normalization and $|V|$ the vocabulary size:

$$
\text{logits} = W_U\,\text{LN}_f(h_L)\in\mathbb{R}^{|V|},\qquad p=\text{softmax}(\text{logits})
$$

let's break this formula step-by-step:
- $\text{LN}_f(h_L)$ is the final stream, normalized once before the output.
- each row of $W_U$ is a direction associated with one token, and the logit of this token is the dot product between the stream and this row: the more the stream points toward the row, the more likely the token.
- the softmax turns the logits into next-token probabilities.

## Usage

- **Reading a direction (logit lens)**: for a direction $v$ in the stream space, $W_U v$ gives one value per token. The tokens with the highest values are those the direction promotes, the lowest those it suppresses.
- **Caveat**: this reading is exact only at the last layer. For earlier layers it is an approximation, because later blocks still modify the stream, and it ignores the final normalization.
