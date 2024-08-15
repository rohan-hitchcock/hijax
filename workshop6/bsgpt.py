"""
Bitesized GPT character model for next-char-prediction based on the Holmesian
canon.
"""

import functools
import collections

import numpy as np
import jax
import jax.numpy as jnp
import einops
import optax
import equinox as eqx

from jaxtyping import Array, Float, UInt8 as Byte, PRNGKeyArray

import tqdm
import mattplotlib as mp


# # # 
# Architecture


class MultiHeadedCausalSelfAttention(eqx.Module):
    query_map: Float[Array, "embed_size embed_size"]
    key_map: Float[Array, "embed_size embed_size"]
    value_map: Float[Array, "embed_size embed_size"]
    num_heads: int

    def __init__(
        self,
        key: PRNGKeyArray,
        embed_size: int,
        max_context_length: int,
        num_heads: int,
    ):
        # validate dimensions
        if embed_size % num_heads:
            raise ValueError("num_heads must divide embed_size")
        self.num_heads = num_heads

        # batched key/query/value projections
        bound_attn = 1/jnp.sqrt(embed_size)
        q, k, v = jax.random.uniform(
            key=key,
            shape=(3, embed_size, embed_size),
            minval=-bound_attn,
            maxval=+bound_attn,
        )
        self.query_map = q
        self.key_map = k
        self.value_map = v

    def __call__(
        self,
        x: Float[Array, "t embed_size"],
    ):
        # perform query, key, value transformations (on all heads at once)
        q = x @ self.query_map      # t C @ C C -> t C
        k = x @ self.key_map        # "
        v = x @ self.value_map      # "
        
        # reshape the embed dimension into separate heads
        q_perhead = einops.rearrange(q, 't (h c) -> t h c', h=self.num_heads)
        k_perhead = einops.rearrange(k, 't (h c) -> t h c', h=self.num_heads)
        v_perhead = einops.rearrange(v, 't (h c) -> t h c', h=self.num_heads)

        # vmap the attention computation across each head
        def single_head_attention(
            q: Float[Array, "t head_size"],
            k: Float[Array, "t head_size"],
            v: Float[Array, "t head_size"],
        ) -> Float[Array, "t head_size"]:
            t, head_size = q.shape
            # compute raw affinities
            a = (q @ k.T)                               # tq c @ c tk -> tq tk
            # scale and causally mask them
            a = a * 1/jnp.sqrt(head_size)            # tq tk / . . -> tq tk
            a = a + jnp.log(jnp.tril(jnp.ones((t, t)))) # tq tk + t t -> tq tk
            # convert affinities to mixing weights
            p = jax.nn.softmax(a, axis=-1)              # tq tk -> tq prob(tk)
            # mix values for each key
            y = p @ v                           # tq prob(tk) @ tv c -> t c
            return y
        y_perhead = jax.vmap(
            single_head_attention,
            in_axes=(1,1,1),
            out_axes=1,
        )(
            q_perhead,  # -> t h(vmap) c
            k_perhead,  # -> t h(vmap) c
            v_perhead,  # -> t h(vmap) c
        )               # -> t h(vmap) c

        # recombine heads into new embedding dimension
        y = einops.rearrange(y_perhead, 't h c -> t (h c)')
        return y


class MLP(eqx.Module):
    weights1: Float[Array, "num_inputs num_hidden"] 
    biases1: Float[Array, "num_hidden"]
    weights2: Float[Array, "num_hidden num_outputs"]
    biases2: Float[Array, "num_outputs"]

    def __init__(
        self,
        key: PRNGKeyArray,
        num_inputs: int,
        num_hidden: int,
        num_outputs: int,
    ):
        k1, k2 = jax.random.split(key)
        # layer 1
        bound_layer1 = 1/jnp.sqrt(num_inputs)
        self.weights1 = jax.random.uniform(
            key=k1,
            shape=(num_inputs, num_hidden),
            minval=-bound_layer1,
            maxval=+bound_layer1,
        )
        self.biases1 = jnp.zeros(num_hidden)
        # layer 2
        bound_layer2 = 1/jnp.sqrt(num_hidden)
        self.weights2 = jax.random.uniform(
            key=k2,
            shape=(num_hidden, num_outputs),
            minval=-bound_layer2,
            maxval=+bound_layer2,
        )
        self.biases2 = jnp.zeros(num_outputs)

    def __call__(
        self,
        x: Float[Array, "num_inputs"],
    ) -> Float[Array, "num_outputs"]:
        x = x @ self.weights1 + self.biases1
        x = jax.nn.relu(x)
        x = x @ self.weights2 + self.biases2
        return x


class LayerNorm(eqx.Module):
    learned_loc: Float[Array, "size"]
    learned_scale: Float[Array, "size"]
    
    def __init__(self, size: int):
        self.learned_loc = jnp.zeros(size)
        self.learned_scale = jnp.ones(size)

    def __call__(self, x: Float[Array, "size"]) -> Float[Array, "size"]:
        x_norm = (x - jnp.mean(x)) * 1/jnp.sqrt(jnp.var(x) + 1e-5)
        return x_norm * self.learned_scale + self.learned_loc


class MultiHeadedCausalSelfAttentionBlock(eqx.Module):
    attention: MultiHeadedCausalSelfAttention
    compute: MLP
    pre_attn_ln: LayerNorm
    pre_mlp_ln: LayerNorm
    
    def __init__(
        self,
        key: PRNGKeyArray,
        embed_size: int,
        mlp_size: int,
        max_context_length: int,
        num_heads: int,
    ):
        k1, k2 = jax.random.split(key)
        # attention
        self.pre_attn_ln = LayerNorm(size=embed_size)
        self.attention = MultiHeadedCausalSelfAttention(
            key=k1,
            embed_size=embed_size,
            max_context_length=max_context_length,
            num_heads=num_heads,
        )
        # compute
        self.pre_mlp_ln = LayerNorm(size=embed_size)
        self.compute = MLP(
            key=k2,
            num_inputs=embed_size,
            num_hidden=mlp_size,
            num_outputs=embed_size,
        )

    def __call__(
        self,
        x: Float[Array, "t embed_size"],
    ) -> Float[Array, "t embed_size"]:
        x = x + self.attention(self.pre_attn_ln(x))
        x = x + jax.vmap(self.compute)(jax.vmap(self.pre_mlp_ln)(x))
        return x


class DecodeTransformer(eqx.Module):
    token_embedding: Float[Array, "alphabet_size embed_size"]
    postn_embedding: Float[Array, "max_context_len embed_size"]
    blocks: list[MultiHeadedCausalSelfAttentionBlock]
    unembedding_ln: LayerNorm
    unembedding: Float[Array, "embed_size alphabet_size"]
    
    def __init__(
        self,
        key: PRNGKeyArray,
        alphabet_size: int,
        max_context_length: int,
        embed_size: int,
        mlp_size: int,
        num_heads: int,
        num_blocks: int,
    ):
        k1, k2, k3, k4 = jax.random.split(key, 4)
        bound_embed = 1/jnp.sqrt(alphabet_size)
        self.token_embedding = jax.random.uniform(
            key=k1,
            shape=(alphabet_size, embed_size),
            minval=-bound_embed,
            maxval=+bound_embed,
        )
        self.postn_embedding = jax.random.uniform(
            key=k2,
            shape=(max_context_length, embed_size),
            minval=-bound_embed,
            maxval=+bound_embed,
        )
        self.blocks = [
            MultiHeadedCausalSelfAttentionBlock(
                key=k_block,
                embed_size=embed_size,
                mlp_size=mlp_size,
                max_context_length=max_context_length,
                num_heads=num_heads,
            )
            for k_block in jax.random.split(k3, num_blocks)
        ]
        self.unembedding_ln = LayerNorm(embed_size)
        bound_unembed = 1/jnp.sqrt(embed_size)
        self.unembedding = jax.random.uniform(
            key=k4,
            shape=(embed_size, alphabet_size),
            minval=-bound_unembed,
            maxval=+bound_unembed,
        )

    def __call__(
        self,
        tokens: Float[Array, "t alphabet_size"],
    ) -> Float[Array, "t alphabet_size"]:
        t, _v = tokens.shape
        T_max, _C = self.postn_embedding.shape
        if t > T_max:
            raise ValueError(f"too many tokens! {t} > {T_max}")

        # semantic and positional token embeddings
        x_semantics = tokens @ self.token_embedding     # t v @ v C -> t C
        x_positions = self.postn_embedding[:t, :]       #   T_max C -> t C
        x = x_semantics + x_positions                   # t C + t C -> t C

        # apply the num_blocks attention blocks in sequence
        for block in self.blocks:
            x = x + block(x)                            # t C + t C -> t C

        # unembedding: transform back to predicted next token probs
        x = self.unembedding_ln(x)                      # t C -> t C
        logits = x @ self.unembedding                   # t C @ C v -> t v
        probs = jax.nn.softmax(logits, axis=-1)         # t v -> t prob(v)
        return probs
    

class ByteSequenceModel(eqx.Module):
    decode_transformer: DecodeTransformer
    max_context_length: int
    
    def __init__(
        self, 
        key: PRNGKeyArray,
        max_context_length: int,
        embed_size: int,
        mlp_size: int,
        num_heads: int,
        num_blocks: int,
    ):
        self.max_context_length = max_context_length
        self.decode_transformer = DecodeTransformer(
            key=key,
            alphabet_size=128,
            max_context_length=max_context_length,
            embed_size=embed_size,
            mlp_size=mlp_size,
            num_heads=num_heads,
            num_blocks=num_blocks,
        )

    def forward(
        self,
        byte_array: Byte[Array, "t"],
    ) -> Float[Array, "t 128"]:
        tokens_one_hot = jnp.eye(128)[byte_array]
        prob_next_tokens = self.decode_transformer(tokens_one_hot)
        return prob_next_tokens

    def forward_batch(
        self,
        byte_arrays: Byte[Array, "b t"],
    ) -> Float[Array, "b t 128"]:
        return jax.vmap(self.forward)(byte_arrays)

    def complete(
        self,
        key: PRNGKeyArray,
        prompt_tokens: Byte[Array, "num_tokens_in"],
        num_tokens_out: int,
    ) -> Byte[Array, "num_tokens_out"]:
        # set up a buffer
        num_tokens_in, = prompt_tokens.shape
        buffer = jnp.concatenate((
            prompt_tokens,
            jnp.zeros(num_tokens_out, dtype=jnp.uint8),
        ))
        # autoregressive loop
        for i in range(num_tokens_out):
            hi = num_tokens_in + i
            lo = max(0, hi - self.max_context_length)
            prob_next_token = self.forward(buffer[lo:hi])[-1]
            key_next_token, key = jax.random.split(key)
            # next_token = jnp.argmax(prob_next_token).astype(jnp.uint8)
            next_token = jax.random.choice(
                key=key_next_token,
                a=128,
                p=prob_next_token,
                shape=(),
            ).astype(dtype=jnp.uint8)
            buffer = buffer.at[hi].set(next_token)
        return buffer[num_tokens_in:]
    

def count_params(tree: eqx.Module) -> int:
    return jax.tree.reduce(
        lambda carry, leaf: carry + (leaf.size if eqx.is_array(leaf) else 0),
        tree,
        0,
    )


# # # 
# Helper functions


def str_to_array(s: str) -> Byte[Array, "len(s)"]:
    return jnp.array([ord(c) for c in s], dtype=jnp.uint8)


def array_to_str(a: Byte[Array, "len(s)"]) -> str:
    return "".join(chr(i) for i in a)


# # # 
# Training loop


def main(
    seed: int = 221,
    max_context_length: int = 32,
    learning_rate: float = 0.001,
    num_steps: int = 512,
    batch_size: int = 32,
    embed_size: int = 64,
    mlp_size: int = 64,
    num_heads: int = 4,
    num_blocks: int = 6,
    steps_per_visualisation: int = 8,
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
    

    print("initialising model...")
    key_model, key = jax.random.split(key)
    model = ByteSequenceModel(
        key=key_model,
        max_context_length=max_context_length,
        embed_size=embed_size,
        mlp_size=mlp_size,
        num_heads=num_heads,
        num_blocks=num_blocks,
    )
    print("  number of parameters:", count_params(model))
    key_completion, key = jax.random.split(key)
    completion = model.complete(
        key=key,
        prompt_tokens=prefix,
        num_tokens_out=max_context_length,
    )
    print("  sample completion:", repr(array_to_str(completion)))


    print("initialising optimiser...")
    optimiser = optax.adam(learning_rate)
    opt_state = optimiser.init(model)
    

    print("initialising eval data...")
    key_eval_batch, key = jax.random.split(key)
    eval_batch_ids = jax.random.choice(
        key=key_eval_batch,
        a=data_test.size - max_context_length,
        shape=(batch_size, 1),
    ) + jnp.arange(max_context_length+1)
    eval_data_batch = data_test[eval_batch_ids]


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
    metrics = collections.defaultdict(list)
    for step in tqdm.trange(num_steps, dynamic_ncols=True):
        # sample a batch of sequences
        key_batch, key = jax.random.split(key)
        batch_ids = jax.random.choice(
            key=key_batch,
            a=data_train.size - max_context_length,
            shape=(batch_size, 1),
        ) + jnp.arange(max_context_length+1)
        data_batch = data_train[batch_ids]

        # compute the batch loss and grad
        (train_loss, ts_pred_distr), grads = eqx.filter_value_and_grad(
            batch_sequence_cross_entropy,
            has_aux=True,
        )(
            model,
            data_batch,
        )

        # compute update, update optimiser and model
        updates, opt_state = optimiser.update(grads, opt_state, model)
        model = eqx.apply_updates(model, updates)

        # compute metrics
        test_loss, _ = batch_sequence_cross_entropy(model, eval_data_batch)
        unigram_score = batch_unigram_cross_entropy(
            unigram_stats=unigram_stats,
            ts_pred_distr=ts_pred_distr,
        )
        bigram_score = batch_bigram_cross_entropy(
            bigram_stats=bigram_stats,
            ts=data_batch,
            ts_pred_distr=ts_pred_distr,
        )
        metrics['train cross entropy (train batch)'].append((step, train_loss))
        metrics['test cross entropy (fixed batch)'].append((step, test_loss))
        metrics['unigram cross entropy (train batch)'].append((step, unigram_score))
        metrics['bigram cross entropy (train batch)'].append((step, bigram_score))

        # visualisation:
        if step % steps_per_visualisation == 0 or step == num_steps - 1:
            key_completion, key = jax.random.split(key)
            example_plot = vis_examples(
                model=model,
                key=key_completion,
                prompts=[
                    "Holmes said \"Elementary, my dear",
                    ". Sherlock Holmes and Doctor Wat",
                ],
            )
            metrics_plot = vis_metrics(metrics, num_steps)
            plot = example_plot ^ metrics_plot
            if step == 0:
                tqdm.tqdm.write(str(plot))
            else:
                tqdm.tqdm.write(f"\x1b[{plot.height}A{plot}")


# # # 
# Loss functions


def batch_sequence_cross_entropy(
    model: ByteSequenceModel,
    ts_batch: Byte[Array, "b t"],
) -> tuple[float, Float[Array, "b t v"]]:
    sequence_cross_entropies, ts_pred_distr = jax.vmap(
        sequence_cross_entropy,
        in_axes=(None, 0),
        out_axes=(0, 0),
    )(
        model,
        ts_batch,
    )
    return sequence_cross_entropies.mean(), ts_pred_distr


def sequence_cross_entropy(
    model: ByteSequenceModel,
    ts: Byte[Array, "t"],
) -> tuple[float, Float[Array, "t v"]]:
    # model outputs for each token
    ts_pred_distr = model.forward(ts[:-1])
    # autoregressive ground truth labels
    ts_true = ts[1:]
    # average per token cross entropy from true tokens
    per_token_cross_entropy = jax.vmap(cross_entropy_dirac)(
        ts_true,
        ts_pred_distr,
    )
    return per_token_cross_entropy.mean(), ts_pred_distr


def cross_entropy_dirac(
    t_true: Byte[Array, ""],
    t_pred_distr: Float[Array, "v"],
) -> float:
    return -jnp.log(t_pred_distr[t_true])


def cross_entropy_distr(
    t_comp_distr: Float[Array, "v"],
    t_pred_distr: Float[Array, "v"],
) -> float:
    return -(t_comp_distr @ jnp.log(t_pred_distr))
        

def batch_unigram_cross_entropy(
    unigram_stats: Float[Array, "v"],
    ts_pred_distr: Float[Array, "b t v"],
) -> float:
    # vmap over t axis of ts_pred_distr
    vmap_ce = jax.vmap(cross_entropy_distr, in_axes=(None, 0))
    # vmap over b axis of ts_pred_distr
    vvmap_ce = jax.vmap(vmap_ce, in_axes=(None, 0))
    # compute the average cross entropy from unigrams
    return vvmap_ce(unigram_stats, ts_pred_distr).mean()


def batch_bigram_cross_entropy(
    bigram_stats: Float[Array, "v v"],
    ts: Byte[Array, "b t+1"],
    ts_pred_distr: Float[Array, "b t v"],
) -> float:
    # select bigram distributions active for each token
    ts_cond_distr = bigram_stats[ts[:,:-1], :]
    # vmap cross entropy function over batch/time axes
    vvmap_ce = jax.vmap(jax.vmap(cross_entropy_distr))
    # compute the average
    return vvmap_ce(ts_cond_distr, ts_pred_distr).mean()


# # # 
# Computing n-gram statistics


def compute_unigram_stats(
    data: Byte[Array, "n"],
    epsilon: float = 1e-7,
) -> Float[Array, "128"]:
    counts = jnp.zeros(128).at[data].add(1) + epsilon
    distr = counts / counts.sum()
    return distr


def compute_conditional_bigram_stats(
    data: Byte[Array, "n"],
    epsilon: float = 1e-7,
) -> Float[Array, "128 128"]:
    counts = jnp.zeros((128, 128)).at[data[:-1], data[1:]].add(1) + epsilon
    distr = counts / counts.sum(axis=1, keepdims=True)
    return distr


# # # 
# Visualisation


def vis_examples(
    model: ByteSequenceModel,
    key: PRNGKeyArray,
    prompts: list[str],
) -> mp.plot:
    plots = []
    for prompt in prompts: # TODO: vmap instead?
        key_prompt, key = jax.random.split(key)
        completion = model.complete(
            key=key_prompt,
            prompt_tokens=str_to_array(prompt),
            num_tokens_out=32,
        )
        render_prompt = repr(prompt)[1:-1] # strip quotes
        render_completion = repr(array_to_str(completion))[1:-1]
        render_example = f"{render_prompt}\n    -> {render_completion}"
        plots.append(mp.text(render_example))
    return mp.border(mp.vstack(
        mp.center(mp.text("example prompt -> completions"), width=78),
        *plots,
    ))


def vis_metrics(
    metrics: dict[str, list[tuple[int, float]]],
    num_steps: int,
) -> mp.plot:
    plots = []
    for metric_name, metric_data in metrics.items():
        data = np.array(metric_data)
        desc = (
            f"min: {data[:,1].min():.3f} | "
            f"max: {data[:,1].max():.3f} | "
            f"last: {data[-1,1]:.3f}"
        )
        plot = mp.border(mp.vstack(
            mp.center(mp.text(metric_name), width=38),
            mp.scatter(
                data=data,
                xrange=(0, num_steps-1),
                yrange=(0, max(6, data[:,1].max())),
                color=(0.2, 1.0, 0.8),
                width=38,
                height=9,
            ),
            mp.text(desc),
        ))
        plots.append(plot)
    return mp.wrap(*plots, cols=2)


if __name__ == "__main__":
    import tyro
    tyro.cli(main)
