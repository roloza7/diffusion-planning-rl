# Latent Diffusion for Planning

## Key Concepts

#### [Forced Diffusion](https://arxiv.org/pdf/2407.01392)

Essentially, this is a diffusion paradigm where a model is given a set of n-tokens and asked to denoise them with _independent_ noise levels.

This is the backbone of the idea, and two of our key assumptions follow:

**We can model the stochasticity of future states by controlling noise levels.**
(i.e.: future states are more noisy, so the model should rely on them less.)

And,

**We can prevent information leakage by cranking up the noise**
(i.e.: our causal model does NOT need masking, because training tokens with independent noise levels means that at inference time masked tokens are instead pure noise, which a well generalized model would in principle identify as unreliable)

#### [Classifier Guidance](https://arxiv.org/pdf/2105.05233)

The less popular cousin of classifier-free guidance, this method trains a classifier and uses its gradients to "nudge" diffusion into specific directions. Usually this is bad because we would need to train an extra classifier, but a world model *already* needs prediction networks to be useful (observation, reward, terminal predictions, etc...)

So we can use this for free, without any need for training beyond what we would have to do anyway. This leads to one of the key hypotheses:

**We can use classifier guidance to nudge tokens to predict specific states**
(i.e.: apply classifier guidance wrt. the gradient of the reward predictor on the n-th token -> n-th denoised token is biased towards a more desirable state)

## What this repo is doing on top of that

Given that:
1. The model doesn't need to be masked autoregressive
2. We can bias independent tokens towards specific objectives
3. The model uses temporal information from other states to reduce state noise

The gamble is that, given a well-trained model, we can formulate an input at infrence time:

1. Take timesteps (t<sub>-ctx</sub>, ..., t<sub>0</sub>) as context timesteps
2. Insert a pure noise token t<sub>n</sub> with a position embedding representing some distant future
3. Insert pure noise tokens (t<sub>1</sub>, ..., t<sub>n-1</sub>) with position embeddings representing intervals between the 'now' and the 'distant future'

By carefully scheduling our noise, we may be able to:

1. Perform guidance on the n-th token with respect to the reward predictor, so the n-th prediction shows a desirable state
2. Allow the in-between noise tokens to settle around t<sub>0</sub> and t<sub>n</sub> in order to form a consistent trajectory

That offloads all of the 'search job' to the neural model, which, well, might or might not pay off.

### Challenges

These are things that can make or break this, but if we can make this work, it might be a pretty cool contribution.

1. Making sure the diffusion model is strict with dynamics modelling (i.e.: not predicting walking through walls or other things that don't make sense) *I may have a solution for this in the form of a GAN-like thing, but it might not be a problem*
2. Figuring out a training exploration strategy or something to make it explore and gather dynamics information.
3. Probably more things will show up.

### Conclusion

If we can do this, it's pretty interesting, because it **completely avoids the value assignment problem** by independently deciding on a valuable state just using the returns. If we can make something consistent despite these challenges it creates a training paradigm where we can just throw more compute and VRAM at a RL problem (in the form of longer sequence lengths and deeper denoisers) instead of worrying about reward shaping, controlling critic variance, and whatever else.