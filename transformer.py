import haiku as hk
import jax
from ring import ml


# Define the Transformer model
class Transformer(hk.Module):
    def __init__(self, embed_dim, num_heads, ff_dim, num_layers, output_dim):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.num_layers = num_layers
        self.output_dim = output_dim

    def __call__(self, x):
        # Embed the input
        x = hk.Linear(self.embed_dim)(x)

        # Create multiple transformer layers
        for _ in range(self.num_layers):
            x = self._transformer_layer(x)

        # Project to the output dimension
        x = hk.Linear(self.output_dim)(x)
        return x

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
        )
        return model(x)

    net = ml.RING(forward_factory=lambda lam: transformer_fn)
    net = ml.base.NoGraph_FilterWrapper(net, quat_normalize=True)
    net = ml.base.ScaleX_FilterWrapper(net)
    return net
