"""
Bitesized GPT character model for next-char-prediction based on the Holmesian
canon. Accelerated with jax.jit.
"""

import functools
import collections
import dataclasses

import numpy as np
import jax
import jax.numpy as jnp
import einops
import optax
import equinox as eqx

from jaxtyping import Array, Float, Int, UInt8 as Byte, PRNGKeyArray, PyTree

import tqdm
import mattplotlib as mp


# # # 
# Wrapper


def struct(Class):
    """
    Wrapper that transforms a class into an immutable dataclass that is also
    registered as a JAX PyTree and uses equinox for __str__ and __repr__.
    """
    # wrap class as an immutable Python dataclass
    Dataclass = dataclasses.dataclass(Class, frozen=True)
    # register dataclass as a JAX pytree node
    jax.tree_util.register_dataclass(
        nodetype=Dataclass,
        data_fields=[field.name for field in dataclasses.fields(Dataclass)],
        meta_fields=[],
    )
    # overwrite string render methods to use equinox pretty-printing
    Dataclass.__repr__ = eqx.tree_pformat
    Dataclass.__str__ = eqx.tree_pformat
    return Dataclass


# # # 
# Architecture


@struct
class LinearTransformParams:
    weights: Float[Array, "num_inputs num_outputs"]


class LinearTransform:
    def __init__(self, num_inputs: int, num_outputs: int):
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs

    @functools.partial(jax.jit, static_argnames=["self"])
    def init(self, key: PRNGKeyArray) -> LinearTransformParams:
        bound = jax.lax.rsqrt(jnp.float32(self.num_inputs))
        return LinearTransformParams(
            weights=jax.random.uniform(
                key=key,
                shape=(self.num_inputs, self.num_outputs),
                minval=-bound,
                maxval=+bound,
            ),
        )

    @functools.partial(jax.jit, static_argnames=["self"])
    def __call__(
        self,
        p: LinearTransformParams,
        x: Float[Array, "num_inputs"],
    ) -> Float[Array, "num_outputs"]:
        return x @ p.weights


@struct
class AffineTransformParams:
    weights: Float[Array, "num_inputs num_outputs"]
    biases: Float[Array, "num_outputs"]


class AffineTransform:
    def __init__(self, num_inputs: int, num_outputs: int):
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs

    @functools.partial(jax.jit, static_argnames=["self"])
    def init(self, key: PRNGKeyArray) -> AffineTransformParams:
        bound = jax.lax.rsqrt(jnp.float32(self.num_inputs))
        return AffineTransformParams(
            weights=jax.random.uniform(
                key=key,
                shape=(self.num_inputs, self.num_outputs),
                minval=-bound,
                maxval=+bound,
            ),
            biases=jnp.zeros(self.num_outputs),
        )

    @functools.partial(jax.jit, static_argnames=["self"])
    def __call__(
        self,
        p: AffineTransformParams,
        x: Float[Array, "num_inputs"],
    ) -> Float[Array, "num_outputs"]:
        return x @ p.weights + p.biases


@struct
class MultiHeadedCausalSelfAttentionParams:
    QKV: LinearTransformParams # LinearTransformParams[3]


class MultiHeadedCausalSelfAttention:
    def __init__(self, embed_size: int, num_heads: int):
        self.num_heads = num_heads
        self.qkv_maps = LinearTransform(
            num_inputs=embed_size,
            num_outputs=embed_size,
        )

    @functools.partial(jax.jit, static_argnames=["self"])
    def init(self, key: PRNGKeyArray) -> MultiHeadedCausalSelfAttentionParams:
        keys = jax.random.split(key, 3)
        return MultiHeadedCausalSelfAttentionParams(
            QKV=jax.vmap(self.qkv_maps.init)(keys),
        )

    @functools.partial(jax.jit, static_argnames=["self"])
    def __call__(
        self,
        p: MultiHeadedCausalSelfAttentionParams,
        x: Float[Array, "t embed_size"],
    ) -> Float[Array, "t embed_size"]:
        # perform query, key, value transformations (on all heads at once)
        qkv = jax.vmap(self.qkv_maps, in_axes=(0,None))(p.QKV, x)
        # reshape the embed dimension into separate heads
        qkv_perhead = einops.rearrange(
            qkv,
            'qkv t (num_heads head_size) -> qkv t num_heads head_size',
            num_heads=self.num_heads,
        )
        # vmap the attention computation across each head
        def single_head_attention(
            qkv: Float[Array, "3 t head_size"],
        ) -> Float[Array, "t head_size"]:
            q, k, v = qkv
            t, head_size = q.shape
            # compute raw affinities                tq c @ c tk -> tq tk
            a = (q @ k.T)                                   
            # scale                                 tq tk / . . -> tq tk
            a = a * jax.lax.rsqrt(jnp.float32(head_size))
            # apply causal mask                     tq tk + t t -> tq tk
            a = a + jnp.log(jnp.tril(jnp.ones((t, t))))
            # convert affinities to mixing weights  tq tk -> tq prob(tk)
            p = jax.nn.softmax(a, axis=-1)
            # mix values for each key               tq prob(tk) @ tv c -> t c
            y = p @ v
            return y
        y_perhead = jax.vmap(
            single_head_attention,
            in_axes=2,  # qkv t vmap(num_heads) head_size
            out_axes=1, #     t vmap(num_heads) head_size
        )(qkv_perhead)
        # recombine heads into new embedding dimension
        y = einops.rearrange(
            y_perhead,
            't num_heads head_size -> t (num_heads head_size)',
        )
        return y
    

@struct
class MLPParams:
    layer1: AffineTransformParams
    layer2: AffineTransformParams


class MLP:
    def __init__(self, num_inputs: int, num_hidden: int, num_outputs: int):
        self.layer1 = AffineTransform(num_inputs, num_hidden)
        self.layer2 = AffineTransform(num_hidden, num_outputs)

    @functools.partial(jax.jit, static_argnames=["self"])
    def init(self, key: PRNGKeyArray) -> MLPParams:
        k1, k2 = jax.random.split(key)
        return MLPParams(
            layer1=self.layer1.init(k1),
            layer2=self.layer2.init(k2),
        )

    @functools.partial(jax.jit, static_argnames=["self"])
    def __call__(
        self,
        p: MLPParams,
        x: Float[Array, "num_inputs"],
    ) -> Float[Array, "num_outputs"]:
        x = self.layer1(p.layer1, x)
        x = jax.nn.relu(x)
        x = self.layer2(p.layer2, x)
        return x


@struct
class LayerNormParams:
    loc: Float[Array, "size"]
    scale: Float[Array, "size"]


class LayerNorm:
    def __init__(self, size: int):
        self.size = size
    
    @functools.partial(jax.jit, static_argnames=["self"])
    def init(self, key: PRNGKeyArray) -> LayerNormParams:
        return LayerNormParams(
            loc=jnp.zeros(self.size),
            scale=jnp.ones(self.size),
        )

    @functools.partial(jax.jit, static_argnames=["self"])
    def __call__(
        self, 
        p: LayerNormParams,
        x: Float[Array, "size"],
    ) -> Float[Array, "size"]:
        x_mean = jnp.mean(x)
        x_rstd = jax.lax.rsqrt(jnp.var(x) + 1e-5)
        x_norm = (x - x_mean) * x_rstd
        return x_norm * p.scale + p.loc


@struct
class DecodeTransformerBlockParams:
    layernorm1: LayerNormParams
    attention: MultiHeadedCausalSelfAttentionParams
    layernorm2: LayerNormParams
    compute: MLPParams


class DecodeTransformerBlock:
    def __init__(
        self,
        embed_size: int,
        num_heads: int,
        mlp_size: int,
    ):
        self.attention = MultiHeadedCausalSelfAttention(
            embed_size=embed_size,
            num_heads=num_heads,
        )
        self.compute = MLP(
            num_inputs=embed_size,
            num_hidden=mlp_size,
            num_outputs=embed_size,
        )
        # same layernorm size for both, only need one
        self.layernorm = LayerNorm(size=embed_size)

    @functools.partial(jax.jit, static_argnames=["self"])
    def init(self, key: PRNGKeyArray) -> DecodeTransformerBlockParams:
        k1, k2, k3, k4 = jax.random.split(key, 4)
        return DecodeTransformerBlockParams(
            layernorm1=self.layernorm.init(k1),
            attention=self.attention.init(k2),
            layernorm2=self.layernorm.init(k1),
            compute=self.compute.init(k4),
        )

    @functools.partial(jax.jit, static_argnames=["self"])
    def __call__(
        self,
        p: DecodeTransformerBlockParams,
        x: Float[Array, "t embed_size"],
    ) -> Float[Array, "t embed_size"]:
        # attention between tokens with per-token pre-layernorm
        x_norm = jax.vmap(self.layernorm, in_axes=(None, 0))(p.layernorm1, x)
        x = x + self.attention(p.attention, x_norm)
        # compute with pre-layernorm (both per-token)
        x_norm = jax.vmap(self.layernorm, in_axes=(None, 0))(p.layernorm2, x)
        x = x + jax.vmap(self.compute, in_axes=(None, 0))(p.compute, x_norm)
        return x


@struct
class DecodeTransformerParams:
    token_embedding: LinearTransformParams
    postn_embedding: LinearTransformParams
    blocks: list[DecodeTransformerBlockParams]
    unembedding_layernorm: LayerNormParams
    unembedding: AffineTransformParams


class DecodeTransformer:
    def __init__(
        self,
        alphabet_size: int,
        max_context_length: int,
        embed_size: int,
        mlp_size: int,
        num_heads: int,
        num_blocks: int,
    ):
        self.token_embedding = LinearTransform(alphabet_size, embed_size)
        self.postn_embedding = LinearTransform(max_context_length, embed_size)
        self.transformer_block = DecodeTransformerBlock(
            embed_size=embed_size,
            num_heads=num_heads,
            mlp_size=mlp_size,
        )
        self.num_blocks = num_blocks
        self.unembedding_layernorm = LayerNorm(embed_size)
        self.unembedding = AffineTransform(embed_size, alphabet_size)

    @functools.partial(jax.jit, static_argnames=["self"])
    def init(self, key: PRNGKeyArray) -> DecodeTransformerParams:
        k1, k2, k3, k4, k5 = jax.random.split(key, 5)
        k_blocks = jax.random.split(k3, self.num_blocks)
        return DecodeTransformerParams(
            token_embedding=self.token_embedding.init(k1),
            postn_embedding=self.postn_embedding.init(k2),
            blocks=[
                self.transformer_block.init(k3i)
                for k3i in jax.random.split(k3, self.num_blocks)
            ],
            unembedding_layernorm=self.unembedding_layernorm.init(k4),
            unembedding=self.unembedding.init(k5),
        )

    @functools.partial(jax.jit, static_argnames=["self"])
    def __call__(
        self,
        p: DecodeTransformerParams,
        ts: Float[Array, "t alphabet_size"],
    ) -> Float[Array, "t alphabet_size"]:
        context_length, _alphabet_size = ts.shape

        # embedding: semantic and positional token embeddings
        x_semantic = self.token_embedding(p.token_embedding, ts)
        x_position = p.postn_embedding.weights[:context_length, :]
        x = x_semantic + x_position                         # -> t embed_size
        # apply the num_blocks attention blocks in sequence
        for block_params in p.blocks:
            x = self.transformer_block(block_params, x)     # -> t embed_size
        # unembedding: transform back to predicted next token probs
        x_norm = self.unembedding_layernorm(p.unembedding_layernorm, x)
        logits = self.unembedding(p.unembedding, x_norm)    # -> t vocab
        probs = jax.nn.softmax(logits, axis=-1)             # -> t prob(vocab)
        return probs


@struct
class ByteSequenceModelParams:
    decode_transformer: DecodeTransformerParams
    

class ByteSequenceModel:
    def __init__(
        self, 
        max_context_length: int,
        embed_size: int,
        mlp_size: int,
        num_heads: int,
        num_blocks: int,
    ):
        self.max_context_length = max_context_length
        self.decode_transformer = DecodeTransformer(
            alphabet_size=128,
            max_context_length=max_context_length,
            embed_size=embed_size,
            mlp_size=mlp_size,
            num_heads=num_heads,
            num_blocks=num_blocks,
        )


    @functools.partial(jax.jit, static_argnames=["self"])
    def init(self, key: PRNGKeyArray) -> ByteSequenceModelParams:
        return ByteSequenceModelParams(
            decode_transformer=self.decode_transformer.init(key),
        )


    @functools.partial(jax.jit, static_argnames=["self"])
    def forward(
        self,
        p: ByteSequenceModelParams,
        byte_array: Byte[Array, "t"],
    ) -> Float[Array, "t 128"]:
        tokens_one_hot = jax.nn.one_hot(byte_array, num_classes=128)
        prob_next_tokens = self.decode_transformer(
            p=p.decode_transformer,
            ts=tokens_one_hot,
        )
        return prob_next_tokens


    @functools.partial(jax.jit, static_argnames=["self"])
    def forward_batch(
        self,
        p: ByteSequenceModelParams,
        byte_arrays: Byte[Array, "b t"],
    ) -> Float[Array, "b t 128"]:
        batched = jax.vmap(self.forward, in_axes=(None, 0))
        return batched(p, byte_arrays)


    def complete(
        self,
        p: ByteSequenceModelParams,
        key: PRNGKeyArray,
        prompt_tokens: Byte[Array, "num_tokens_in"],
        num_tokens_out: int,
    ) -> Byte[Array, "num_tokens_out"]:
        # insist on minimum num tokens
        num_tokens_in, = prompt_tokens.shape
        assert num_tokens_in >= self.max_context_length
        # set up buffer we will slide across
        buffer = jnp.concatenate((
            prompt_tokens[-self.max_context_length:],
            jnp.zeros(num_tokens_out, dtype=jnp.uint8),
        ))
        # loop across buffer
        keys_next_token = jax.random.split(key, num_tokens_out)
        los = jnp.arange(num_tokens_out)
        for lo, key_next_token in zip(los, keys_next_token):
            # slice window
            window = buffer[lo:lo+self.max_context_length]
            # predict next token
            prob_next_token = self.forward(p, window)[-1]
            next_token = jax.random.choice(
                key=key_next_token,
                a=128,
                p=prob_next_token,
                shape=(),
            ).astype(dtype=jnp.uint8)
            # add token to buffer
            buffer = buffer.at[lo+self.max_context_length].set(next_token)
        # return completion
        tokens_out = buffer[-num_tokens_out:]
        return tokens_out


# # # 
# Helper functions


def count_params(p: PyTree) -> int:
    return jax.tree.reduce(lambda carry, leaf: carry + leaf.size, p, 0)


def str_to_array(s: str) -> Byte[Array, "len(s)"]:
    return jnp.array([ord(c) for c in s], dtype=jnp.uint8)


def array_to_str(a: Byte[Array, "len(s)"]) -> str:
    return "".join(chr(i) for i in a)


@jax.jit
def compute_unigram_stats(
    data: Byte[Array, "n"],
    epsilon: float = 1e-7,
) -> Float[Array, "128"]:
    counts = jnp.zeros(128).at[data].add(1) + epsilon
    distr = counts / counts.sum()
    return distr


@jax.jit
def compute_conditional_bigram_stats(
    data: Byte[Array, "n"],
    epsilon: float = 1e-7,
) -> Float[Array, "128 128"]:
    counts = jnp.zeros((128, 128)).at[data[:-1], data[1:]].add(1) + epsilon
    distr = counts / counts.sum(axis=1, keepdims=True)
    return distr


# # # 
# Cross entropy functions


@jax.jit
def cross_entropy_dirac(
    true_index: Int[Array, ""],
    pred_distr: Float[Array, "v"],
) -> float:
    return -jnp.log(pred_distr[true_index])


@jax.jit
def cross_entropy_distr(
    true_distr: Float[Array, "v"],
    pred_distr: Float[Array, "v"],
) -> float:
    return -(true_distr @ jnp.log(pred_distr))


@jax.jit
def batch_cross_entropy_distr(
    true_distrs: Float[Array, "... v"],
    pred_distrs: Float[Array, "... v"],
) -> Float[Array, "..."]:
    batched = jnp.vectorize(cross_entropy_distr, signature='(v),(v)->()')
    return batched(true_distrs, pred_distrs)


@jax.jit
def batch_cross_entropy_dirac(
    true_indexs: Int[Array, "..."],
    pred_distrs: Float[Array, "... v"],
) -> Float[Array, "..."]:
    batched = jnp.vectorize(cross_entropy_dirac, signature='(),(v)->()')
    return batched(true_indexs, pred_distrs)
    

# # # 
# Loss function


@functools.partial(jax.jit, static_argnames=["model"])
def loss_fn(
    params: ByteSequenceModelParams,
    model: ByteSequenceModel,
    tokens: Byte[Array, "b t+1"],
) -> float:
    return batch_cross_entropy_dirac(
        true_indexs=tokens[:,1:],
        pred_distrs=model.forward_batch(params, tokens[:,:-1]),
    ).mean()


# # # 
# Training loop


def main(
    seed: int = 221,
    max_context_length: int = 32,
    learning_rate: float = 0.001,
    num_steps: int = 512,
    batch_size: int = 32,
    # architecture details
    embed_size: int = 64,
    mlp_size: int = 64,
    num_heads: int = 4,
    num_blocks: int = 6,
    # training loop details
    num_steps_per_vis: int = 8,
):
    key = jax.random.key(seed)
    

    print("loading byte corpus...")
    with open("sherlock.txt") as file:
        data = str_to_array(file.read())
    data_train = data[:3_200_000]
    data_test = data[3_200_000:]
    print("  number of training tokens (bytes):", *data_train.shape)
    print("  number of testing tokens (bytes): ", *data_test.shape)
    prefix = data_test[:max_context_length]
    print("  test data prefix: ", repr(array_to_str(prefix)))
    

    print("configuring model architecture...")
    model = ByteSequenceModel(
        max_context_length=max_context_length,
        embed_size=embed_size,
        mlp_size=mlp_size,
        num_heads=num_heads,
        num_blocks=num_blocks,
    )
    print(model)


    print("initialising model params...")
    key_model, key = jax.random.split(key)
    params = model.init(key_model)
    print("  number of parameters:", count_params(params))
    print(params)


    print("testing model completion...")
    key_completion, key = jax.random.split(key)
    completion = model.complete(
        key=key,
        p=params,
        prompt_tokens=prefix,
        num_tokens_out=max_context_length,
    )
    print("  sample completion:", repr(array_to_str(completion)))


    print("initialising optimiser...")
    optimiser = optax.adam(learning_rate)
    opt_state = optimiser.init(params)
    # print(opt_state)


    print("initialising eval data...")
    key_eval_batch, key = jax.random.split(key)
    eval_batch_ids = jax.random.choice(
        key=key_eval_batch,
        a=data_test.size - max_context_length,
        shape=(num_steps_per_vis * batch_size, 1),
    ) + jnp.arange(max_context_length+1)
    eval_data_batch = data_test[eval_batch_ids]
    
    
    print("initialising example prompts...")
    example_prompts = jnp.stack([
        str_to_array("Holmes said \"Elementary, my dear"),
        str_to_array(". Sherlock Holmes and Doctor Wat"),
    ])


    print("preparing comparison n-gram distributions...")
    unigram_stats = compute_unigram_stats(data_train)
    unigram_entropy = cross_entropy_distr(unigram_stats, unigram_stats)
    print("  unigram entropy:", unigram_entropy)
    bigram_stats = compute_conditional_bigram_stats(data_train)
    bigram_entropy = unigram_stats @ jax.vmap(
        cross_entropy_distr
    )(bigram_stats, bigram_stats)
    print("  bigram entropy:", bigram_entropy)

 
    print("training loop...")
    train_steps = tqdm.trange(num_steps)
    metrics = collections.defaultdict(list)
    for step in train_steps:
        # sample a batch of sequences
        key_batch, key = jax.random.split(key)
        batch_ids = jax.random.choice(
            key=key_batch,
            a=data_train.size - max_context_length,
            shape=(batch_size, 1),
        ) + jnp.arange(max_context_length+1)
        data_batch = data_train[batch_ids]
        
        # compute the batch loss and grad
        train_loss, grads = jax.value_and_grad(loss_fn)(
            params,
            model,
            data_batch,
        )
        
        # compute update, update optimiser and model
        updates, opt_state = optimiser.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        
        metrics['train loss'].append((step, train_loss))

        if step % num_steps_per_vis == 0:
            # periodically compute additional metrics
            ts_pred_distr = model.forward_batch(params, eval_data_batch[:,:-1])
            test_loss = batch_cross_entropy_dirac(
                true_indexs=eval_data_batch[:,1:],
                pred_distrs=ts_pred_distr,
            ).mean()
            unigram_score = batch_cross_entropy_distr(
                true_distrs=unigram_stats,
                pred_distrs=ts_pred_distr,
            ).mean()
            bigram_score = batch_cross_entropy_distr(
                true_distrs=bigram_stats[eval_data_batch[:,:-1]],
                pred_distrs=ts_pred_distr,
            ).mean()
            end_step = step + num_steps_per_vis - 1
            metrics['test loss'].append((end_step, test_loss))
            metrics['unigram score'].append((end_step, unigram_score))
            metrics['bigram score'].append((end_step, bigram_score))
        
            # periodically compute examples
            key_completion, key = jax.random.split(key)
            example_completions = jax.vmap(
                model.complete,
                in_axes=(None,0,0,None)
            )(
                params,
                jax.random.split(key_completion, 2),
                example_prompts,
                32,
            )
            
            # periodically update visualisation
            plot = (
                vis_examples(example_prompts, example_completions)
                ^ vis_metrics(metrics, total=num_steps)
            )
            if step == 0:
                tqdm.tqdm.write(str(plot))
            else:
                tqdm.tqdm.write(f"\x1b[{plot.height}A{plot}")


# # # 
# Visualisation


def vis_metrics(
    metrics: dict[str, tuple[int, float]],
    total: int,
) -> mp.plot:
    plots = []
    for metric_name, metric_data in metrics.items():
        data = np.array(metric_data)
        xs = data[:,1]
        description = (
            f"min: {xs.min():.3f} | max: {xs.max():.3f} | last: {xs[-1]:.3f}"
        )
        plot = mp.border(mp.vstack(
            mp.center(mp.text(metric_name), width=38),
            mp.scatter(
                data=data,
                xrange=(0, total-1),
                yrange=(0, max(6, xs.max())),
                color=(0.2, 1.0, 0.8),
                width=38,
                height=9,
            ),
            mp.text(description),
        ))
        plots.append(plot)
    return mp.wrap(*plots, cols=2)


def vis_examples(
    prompts: Byte[Array, "num_examples num_tokens_in"],
    completions: Byte[Array, "num_examples num_tokens_out"],
) -> mp.plot:
    plots = []
    for prompt, completion in zip(prompts, completions):
        render_prompt = repr(array_to_str(prompt))[1:-1] # strip quotes
        render_completion = repr(array_to_str(completion))[1:-1]
        render_example = f"[{render_prompt}]\n    -> [{render_completion}]"
        plots.append(mp.text(render_example))
    return mp.border(mp.vstack(
        mp.center(mp.text("[example prompt] -> [completions]"), width=78),
        *plots,
    ))


if __name__ == "__main__":
    import tyro
    tyro.cli(main)
