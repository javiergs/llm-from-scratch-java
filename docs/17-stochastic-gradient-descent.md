# Lesson 17: Stochastic gradient descent (SGD)

The previous lessons introduced gradient calculations and a way to replace model parameters. **SGD** uses those gradients to move weights in a direction that locally reduces loss.

For each parameter matrix:

```text
newWeights = oldWeights - learningRate * gradients
```

A learning rate that is too large can make optimization unstable; one that is too small can make learning slow.

## Using the optimizer

```java
StochasticGradientDescent optimizer = new StochasticGradientDescent(0.01);
INDArray[] gradients = /* derivatives in model.parameters() order */;
optimizer.step(model, gradients);
```

The `gradients` array is **not yet generated automatically** by `GptModel`. A later lesson will connect the existing backward rules to produce it from next-token prediction loss.

The optimizer checks gradient shapes and non-finite values before changing the model. Parameter arrays returned by `model.parameters()` are defensive copies.

## Verify

Run `mvn verify`. Tests verify the update equation, a decrease in a simple quadratic objective, model parameter changes, and rejection of malformed gradients without partial updates.

## Scope

SGD is intentionally minimal: no momentum, Adam, learning-rate scheduling, or automatic differentiation. **This lesson does not yet train the GPT model on text.**
