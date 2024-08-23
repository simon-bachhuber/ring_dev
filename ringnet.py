from types import SimpleNamespace
from typing import Callable, Optional

import flax.linen as nn
import haiku as hk
import haiku.experimental.flax as hkflax
import jax
import jax.numpy as jnp
from ring.maths import safe_normalize
import tree_utils

from slstm import sLSTM


def _scan_sys(lam: list[int], f):
    ys = []
    for i, p in enumerate(lam):
        ys.append(f(i, p))
    return tree_utils.tree_batch(ys, backend="jax")


def _make_rnno_cell_apply_fn(
    lam: list[int],
    inner_cell,
    send_msg,
    send_output,
    hidden_state_dim,
    message_dim,
    output_transform: Callable,
):
    N = len(lam)
    parent_array = jnp.array(lam, dtype=jnp.int32)

    def _rnno_cell_apply_fn(inputs, prev_state):
        empty_message = jnp.zeros((1, message_dim))
        mailbox = jnp.repeat(empty_message, N, axis=0)

        # message is sent using the hidden state of the last cell
        # for LSTM `prev_state` is of shape (2 * hidden_state_dim) du to cell state
        prev_last_hidden_state = prev_state[:, -1, :hidden_state_dim]

        msg = jnp.concatenate(
            (jax.vmap(send_msg)(prev_last_hidden_state), empty_message)
        )

        def accumulate_message(link):
            return jnp.sum(
                jnp.where(
                    jnp.repeat((parent_array == link)[:, None], message_dim, axis=-1),
                    msg[:-1],
                    mailbox,
                ),
                axis=0,
            )

        mailbox = jax.vmap(accumulate_message)(jnp.arange(N))

        def cell_input(i: int, p: int):
            local_input = inputs[i]
            local_cell_input = tree_utils.batch_concat_acme(
                (local_input, msg[p], mailbox[i]), num_batch_dims=0
            )
            return local_cell_input

        stacked_cell_input = _scan_sys(lam, cell_input)

        def update_state(cell_input, state):
            cell_output, state = inner_cell(cell_input, state)
            output = output_transform(send_output(cell_output))
            return output, state

        y, state = jax.vmap(update_state)(stacked_cell_input, prev_state)
        return y, state

    return _rnno_cell_apply_fn


def make_ring(
    lam: list[int],
    message_dim: int = 200,
    head_dim: int = 16,
    head_num: int = 8,
    stack_rnn_cells: int = 2,
    send_message_n_layers: int = 1,
    link_output_dim: int = 4,
    link_output_normalize: bool = True,
    link_output_transform: Optional[Callable] = None,
    layernorm: bool = True,
) -> SimpleNamespace:

    if link_output_normalize:
        assert link_output_transform is None
        link_output_transform = safe_normalize
    else:
        if link_output_transform is None:
            link_output_transform = lambda x: x

    @hk.without_apply_rng
    @hk.transform_with_state
    def forward(X):
        # to_latent = hk.Linear(latent_dim)
        # X = to_latent(X)
        hidden_state_dim = 4 * head_dim * head_num
        inp_dim = 2 * message_dim + X.shape[-1]

        send_msg = hk.nets.MLP(
            [hidden_state_dim] * send_message_n_layers + [message_dim]
        )
        stacked_rnn_cell = StackedRNNCell(
            n_stacks=stack_rnn_cells,
            layernorm=layernorm,
            inp_dim=inp_dim,
            head_dim=head_dim,
            head_num=head_num,
        )
        inner_cell = hkflax.lift(
            stacked_rnn_cell,
            name="flax_stacked_slstm",
        )
        send_output = hk.nets.MLP([inp_dim, link_output_dim])

        def init_fn(*args, **kwargs):
            flat_s = stacked_rnn_cell.init_hidden(
                stacked_rnn_cell.n_stacks,
                stacked_rnn_cell.head_dim,
                stacked_rnn_cell.head_num,
            )
            return jnp.repeat(flat_s[None], len(lam), axis=0)

        state = hk.get_state(
            "inner_cell_state",
            shape=(),
            dtype=jnp.float32,
            init=init_fn,
        )

        y, state = hk.dynamic_unroll(
            _make_rnno_cell_apply_fn(
                lam=lam,
                inner_cell=inner_cell,
                send_msg=send_msg,
                send_output=send_output,
                hidden_state_dim=hidden_state_dim,
                message_dim=message_dim,
                output_transform=link_output_transform,
            ),
            X,
            state,
        )
        hk.set_state("inner_cell_state", state)
        return y

    return forward


class StackedRNNCell(nn.Module):
    n_stacks: int
    layernorm: bool
    inp_dim: int
    head_dim: int
    head_num: int
    ker_size: int = 4
    p_factor: float = 4 / 3

    def setup(self) -> None:
        self.cells = [
            sLSTM(
                self.inp_dim,
                self.head_dim,
                self.head_num,
                ker_size=self.ker_size,
                p_factor=self.p_factor,
            )
            for _ in range(self.n_stacks)
        ]
        self._layernorm = nn.LayerNorm()

    def __call__(self, x, state):
        output = x
        next_state = []
        for i in range(len(self.cells)):
            output, next_state_i = self.cells[i](output, state[i])
            next_state.append(next_state_i)

            if self.layernorm:
                output = self._layernorm(output)

        return output, jnp.stack(next_state)

    @staticmethod
    def init_hidden(n_stacks, head_dim, head_num):
        state = sLSTM.init_hidden(head_dim, head_num, flat=True)
        return jnp.repeat(state[None], n_stacks, axis=0)
