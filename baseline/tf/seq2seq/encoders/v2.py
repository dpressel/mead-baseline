from baseline.tf.tfy import *
from baseline.tf.embeddings import *
from baseline.tf.transformer import transformer_encoder_stack
from baseline.utils import exporter, MAGIC_VARS
from baseline.model import register_encoder
from collections import namedtuple
from eight_mile.tf.layers import *

RNNEncoderOutput = namedtuple("RNNEncoderOutput", ("output", "hidden", "src_mask"))


def _make_src_mask(output, lengths):
    T = output.shape[1]
    src_mask = tf.cast(tf.sequence_mask(lengths, T), dtype=tf.uint8)
    return src_mask


@register_encoder(name='default')
class RNNEncoder(tf.keras.layers.Layer):

    def __init__(self, dsz=None, hsz=None, rnntype='blstm', layers=1, pdrop=0.5, residual=False, create_src_mask=True, name='encoder', scope="RNNEncoder", **kwargs):
        super().__init__(name=name)
        self.residual = residual
        hidden = hsz if hsz is not None else dsz
        Encoder = LSTMEncoderAll if rnntype == 'lstm' else BiLSTMEncoderAll
        self.rnn = Encoder(dsz, hidden, layers, pdrop, name=scope)
        self.src_mask_fn = _make_src_mask if create_src_mask is True else lambda x, y: None

    def call(self, inputs):
        btc, lengths = inputs
        output, hidden = self.rnn((btc, lengths))
        return RNNEncoderOutput(output=output + btc if self.residual else output,
                                hidden=hidden,
                                src_mask=self.src_mask_fn(output, lengths))


TransformerEncoderOutput = namedtuple("TransformerEncoderOutput", ("output", "src_mask"))


@register_encoder(name='transformer')
class TransformerEncoderWrapper(tf.keras.layers.Layer):

    def __init__(self, dsz, hsz=None, num_heads=4, layers=1, dropout=0.5, name='encoder', scope='TransformerEncoder', **kwargs):
        super().__init__(name=name)
        if hsz is None:
            hsz = dsz
        self.proj = tf.keras.layers.Dense(hsz) if hsz != dsz else self._identity
        d_ff = int(kwargs.get('d_ff', 4 * hsz))
        rpr_k = kwargs.get('rpr_k')
        d_k = kwargs.get('d_k')
        activation = kwargs.get('activation', 'relu')
        scale = bool(kwargs.get('scale', True))
        layer_drop = float(kwargs.get('layer_drop', 0.0))
        self.transformer = TransformerEncoderStack(num_heads, d_model=hsz, d_ff=d_ff,
                                                   pdrop=dropout, scale=scale, layers=layers,
                                                   rpr_k=rpr_k, d_k=d_k, activation=activation, layer_drop=layer_drop,
                                                   scope=scope)


    def _identity(self, x):
        return x

    def call(self, inputs):
        bth, lengths = inputs
        T = get_shape_as_list(bth)[1]
        src_mask = tf.sequence_mask(lengths, T, dtype=tf.float32)
        shp = get_shape_as_list(src_mask)
        new_shp = [shp[0]] + [1, 1] + shp[1:]
        src_mask = tf.reshape(src_mask, new_shp)

        bth = self.proj(bth)
        output = self.transformer((bth, src_mask))
        return TransformerEncoderOutput(output=output, src_mask=src_mask)
