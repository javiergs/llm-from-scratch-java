# Lesson 15: Transformer block backward pass

A GPT-style **pre-normalization transformer block** contains two residual branches:

```text
n1 = layerNorm(x)
a  = multiHeadAttention(n1)
r  = x + a
n2 = layerNorm(r)
f  = feedForward(n2)
y  = r + f
```

The backward pass reverses these operations. For each addition, the incoming gradient flows into **both** branches.

```text
df  = dy
dr  = dy + layerNormBackward(r, feedForwardBackward(n2, df).inputGradient)
dx  = dr + layerNormBackward(x, attentionBackward(n1, dr).inputGradient)
```

The attention and feed-forward backward calculations also produce parameter gradients. This lesson composes the backward rules from Lessons 11, 12, and 14; it does **not** update any parameters.

## Verify

Run `mvn verify`. A test checks that gradients propagate through both residual branches and that zero projection matrices reduce the block's input gradient to the upstream gradient.

## Scope

`TransformerBlockGradients` receives explicit weight matrices to make the chain rule visible without mutating model state. [Lesson 16](16-trainable-model-parameters.md) exposes parameter copies and replacement, while [Lessons 17–19](17-stochastic-gradient-descent.md) cover optimization and training.
