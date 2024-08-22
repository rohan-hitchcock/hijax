Workshop 7: Hi, loop acceleration!
===============================================================================

Preliminaries:

* Any questions from last week?
* Installations: nothing new
* Data: copy from last week

Today's workshop files:

```
workshop7/
├── README.md             # (you are here)
├── sherlock.txt          # (copy from last week)
├── bsgpt_jit.py          # accelerated byte language model (refactored!)
├── bsgpt_scan.py         # we'll accelerate it further with jax.lax.scan
└── mattplotlib.py        # (coming some day to PyPI...)
```

Plan
----

There are three loops in the code that we didn't jit/scan last time:

1. The training loop!
2. The loop in the forward pass, applying the model's six transformer blocks.
3. The next token prediction loop for generating 'completions'.

I did some refactoring compared to last time to make the scanning steps
possible (and otherwise general corrections and improvements) including:

* My solution to last week's challenge 1: Split architecture and parameters.
  No more equinox modules or filtering. Can also jit (and vmap) parameter
  initialisation methods.
* Some changes to completion code: no longer supports short prompts.
* Larger but less frequent evaluations in the training loop.
* Refactored the various batched cross entropy functions, which were getting
  really out of hand! Now there are just four:
  * `cross_entropy_dirac` and `cross_entropy_distr`, taking invidual
    distributions.
  * `batch_cross_entropy_di***` vmapping those using `jnp.vectorize`.
* Various other tweaks.

Loops in JAX
------------

There are several different types of looping computations you might want to
do when programming in JAX.

1. **'Parallel' loop.** Looping over some axis of an array.
    ```
    [ x   x   x   ...   x ]
      ↓   ↓   ↓         ↓
      f   f   f   ...   f
      ↓   ↓   ↓         ↓
    [ y   y   y   ...   y ]
    ```
    We are familiar with this already. We use `jax.vmap` for this.

2. **'Series' loop.** Repeatedly applying a function. Each step can depend on
    the output of the previous step.
    ```
    c → f → c → f → c → f → c → ... → c → f → c
    ```
    This is what we need to accelerate a training loop. There are several
    options for this including:
    * `jax.lax.fori_loop` for terminating after a fixed number of steps.
    * `jax.lax.while_loop` for terminating after a condition is met.

3. **General loop.** Mapping along an axis but *also* each step can depend on
    the output of the previous step.
    ```
      [ x       x       x       ...       x ]
        ↓       ↓       ↓                 ↓
    c → f → c → f → c → f → c → ... → c → f → c
        ↓       ↓       ↓                 ↓
      [ y       y       y       ...       y ]
    ```
    This comes up in for example scanning the application of layers of a deep
    network---`c` is the activations, `x` is the parameters, `y` is unused.
    We use `jax.lax.scan` for this.

The variables `c`, `x`, and `y` don't have to be scalars, they can be arrays
or pytrees of arrays.

Challenge
---------

* Refactor the training loop into a jittable, vmappable function (you can
  remove the visualisation and progress bar).

  Vectorise a sweep over different learning rates to find the best one for
  this model.


