import jax
import jax.numpy as jnp
import random
import tqdm
from typing import Iterable
from copy import copy

from jaxtorch import Module, PRNG, Context, Param
from jaxtorch import nn

class SGD(object):

    def __init__(self, parameters: Iterable[Param]):
        self.parameters = list(parameters)

    def step(self, px, grad, lr):
        new_values = copy(px)
        for p in [param.name for param in self.parameters]:
            new_values[p] = px[p] - grad[p] * lr
        return new_values

def square(x):
    return x*x

# Now implement a xor solver:

class MLP(Module):
    def __init__(self):
        self.layers = nn.Sequential(nn.Linear(2, 3),
                                    nn.Tanh(),
                                    # nn.Dropout(0.9),
                                    nn.Linear(3, 1),
                                    nn.Tanh(),
                                    nn.Linear(1,1))

    def forward(self, cx, x):
        return self.layers(cx, x)

model = MLP()

# XOR
data = [
    ([0, 0], 0),
    ([1, 0], 1),
    ([0, 1], 1),
    ([1, 1], 0)
]

opt = SGD(model.parameters())

rng = PRNG(jax.random.PRNGKey(0))

px = model.init_weights(rng.split())
print(model.state_dict(px))

def loss(px, x, y, key):
    cx = Context(px, key)
    return square(model(cx, x) - y).mean()
loss_grad = jax.jit(jax.value_and_grad(loss))

@jax.jit
def train_step(px, x, y, key, lr):
    v_loss, v_grad = loss_grad(px, x, y, key=key)
    return v_loss, opt.step(px, v_grad, lr=lr)

steps = 1_000_000
pbar = tqdm.trange(1, steps)
for step in pbar:
    xs = []
    ys = []
    for _ in range(1):
        (x, y) = random.choice(data)
        xs.append(jnp.array(x, dtype=jnp.float32))
        ys.append(jnp.array(y, dtype=jnp.float32))
    x = jnp.stack(xs)
    y = jnp.stack(ys)
    v_loss, px = train_step(px, x, y, key=rng.split(), lr=0.01)
    if step % 10_000 == 0:
        pbar.write(f'{step:,}: {v_loss:.6e}')