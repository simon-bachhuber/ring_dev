import haiku as hk
import jax
import jax.numpy as jnp
from ring import ml


# Define the positional encoding function
def positional_encoding(seq_len, embed_dim):
    pos = jnp.arange(seq_len)[:, None]  # Shape (seq_len, 1)
    dim = jnp.arange(embed_dim)[None, :]  # Shape (1, embed_dim)
    angle_rates = 1 / jnp.power(10000, (2 * (dim // 2)) / embed_dim)
    angle_rads = pos * angle_rates  # Shape (seq_len, embed_dim)

    # Apply sine to even indices and cosine to odd indices
    sines = jnp.sin(angle_rads[:, 0::2])
    cosines = jnp.cos(angle_rads[:, 1::2])
    pos_encoding = jnp.concatenate([sines, cosines], axis=-1)
    print(pos_encoding.shape)
    return pos_encoding


# Define the Transformer model
class Transformer(hk.Module):
    def __init__(
        self, embed_dim, num_heads, ff_dim, num_layers, output_dim, pos_encoding
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.num_layers = num_layers
        self.output_dim = output_dim
        self.pos_encoding = pos_encoding

    def __call__(self, x):
        assert x.shape[1] == 1
        assert x.ndim == 3
        x = x[:, 0]

        seq_len = x.shape[0]  # Get the sequence length
        pos_encoding = positional_encoding(
            seq_len, self.embed_dim
        )  # Generate positional encodings

        # Embed the input and add positional encodings
        x = hk.Linear(self.embed_dim)(x)

        if self.pos_encoding:
            x = x + pos_encoding  # Adding positional encoding to the input embeddings

        # Create multiple transformer layers
        for _ in range(self.num_layers):
            x = self._transformer_layer(x)

        # Project to the output dimension
        x = hk.Linear(self.output_dim)(x)
        return x[:, None, :]

    def _transformer_layer(self, x):
        # Multi-head attention with explicit weight initializer
        attn = hk.MultiHeadAttention(
            num_heads=self.num_heads,
            key_size=self.embed_dim // self.num_heads,
            w_init=hk.initializers.VarianceScaling(1.0, "fan_avg", "uniform"),
        )(x, x, x)

        # Ensure attention output matches the input shape
        attn_proj = hk.Linear(self.embed_dim)(attn)

        # Residual connection and layer normalization
        x = hk.LayerNorm(axis=-1, create_scale=True, create_offset=True)(attn_proj + x)

        # Feed-forward network
        ff = hk.Linear(self.ff_dim)(x)
        ff = jax.nn.relu(ff)
        ff = hk.Linear(self.embed_dim)(ff)

        # Residual connection and layer normalization
        x = hk.LayerNorm(axis=-1, create_scale=True, create_offset=True)(ff + x)

        return x


def make_transformer(
    embed_dim: int = 512,
    num_heads: int = 8,
    ff_dim: int = 4 * 512,
    num_layers: int = 2,
    output_dim: int = 8,
    pos_encoding: bool = False,
):
    @hk.without_apply_rng
    @hk.transform_with_state
    def transformer_fn(x):
        model = Transformer(
            embed_dim=embed_dim,
            num_heads=num_heads,
            ff_dim=ff_dim,
            num_layers=num_layers,
            output_dim=output_dim,
            pos_encoding=pos_encoding,
        )
        return model(x)

    net = ml.RING(forward_factory=lambda lam: transformer_fn)
    net = ml.base.NoGraph_FilterWrapper(net, quat_normalize=True)
    net = ml.base.ScaleX_FilterWrapper(net)
    return net
