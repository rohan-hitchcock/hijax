Workshop 6: Just-in-time compilation with JAX (part 2)
-------------------------------------------------------------------------------

Preliminaries:

* Any questions from last week?

* Installations: nothing new

* Download new data:
  ```
  curl https://raw.githubusercontent.com/matomatical/bitesizedGPT/main/data/sherlock-ascii.txt -o sherlock.txt
  ```

Today's workshop files and plan:

```
workshop5/
├── bsgpt.py              # study a byte language model implementation
├── bsgpt_jit.py          # accelerate it with jax.jit and eqx.filter_jit
└── mattplotlib.py
```

Challenge:

* Option 1: Make a version without using `eqx.filter_` functions.
  * Option 1a: Separate architecture and parameter structs.
  * Option 1b: Dataclasses with static fields.
  * Option 1c: Implement your own versions of `filter_jit` etc.
* Option 2: Add dropout modules to the transformer. Statically decide whether
  or not to activate them.
