import numpy as np

import tensorflow as tf
from eight_mile.tf.layers import EmbeddingsStack, BiLSTMEncoderSequence, LSTMEncoderSequence, reload_checkpoint

from baseline.embeddings import register_embeddings, create_embeddings
from eight_mile.utils import Offsets, read_json
from eight_mile.tf.embeddings import TensorFlowEmbeddings
from baseline.tf.embeddings import TensorFlowEmbeddingsModel
from baseline.tf.tfy import reload_embeddings, tf_device_wrapper, create_session
import glob
import os

@register_embeddings(name='tag')
class RNNTaggerEmbeddings(TensorFlowEmbeddings):
    def __init__(self, name='', **kwargs):
        super().__init__(name=name)

    @classmethod
    def create_placeholder(cls, name):
        return tf.compat.v1.placeholder(tf.int32, [None, None, None], name=name)

    @classmethod
    @tf_device_wrapper
    def load(cls, basename, **kwargs):

        state_file = f"{basename}.state"
        _state = read_json(state_file)
        _state['sess'] = kwargs.pop('sess', create_session())
        embeddings_info = _state.pop("embeddings")

        pid = basename.split("-")[-1]
        basepath = os.path.dirname(state_file)
        vf = os.path.join(basepath, f'vocabs-*-{pid}.json')
        fname = list(glob.glob(vf))[0]
        vocab = read_json(fname)

        with _state['sess'].graph.as_default():
            embeddings = reload_embeddings(embeddings_info, basename)
            for k in embeddings_info:
                if k in kwargs:
                    _state[k] = kwargs[k]

            model = cls.create(embeddings, **_state)
            if kwargs.get('init', True):
                model.sess.run(tf.compat.v1.global_variables_initializer())
        model.checkpoint = basename
        model.vocab = vocab
        return model

    def get_vocab(self):
        return self.vocab

    @classmethod
    def create(cls, embeddings, **kwargs):
        """Create the model
        :param embeddings: A `dict` of input embeddings
        :param labels: The output labels for sequence tagging
        :param kwargs: See below
        :Keyword Arguments:

        * *lengths_key* (`str`) -- What is the name of the key that is used to get the full temporal length
        * *dropout* (`float`) -- The probability of dropout
        * *dropin* (`Dict[str, float]`) -- For each feature, how much input to dropout
        * *sess* -- An optional `tf.compat.v1.Session`, if not provided, this will be created
        * *username* -- (`str`) A username, defaults to the name of the user on this machine
        * *variational* -- (`bool`) Should we do variational dropout
        * *rnntype* -- (`str`) -- The type of RNN (if this is an RNN), defaults to 'blstm'
        * *layers* -- (`int`) -- The number of layers to apply on the encoder
        * *hsz* -- (`int`) -- The number of hidden units for the encoder
        * *feature* -- feature name, defaults to word

        :return:
        """
        model = cls()
        model.embeddings = embeddings
        inputs = {}

        model.feature = kwargs.get('feature', 'char')

        if model.feature not in embeddings:
            raise Exception(f"Required {model.feature} input")
        if len(embeddings) > 1:
            raise Exception('We can only use a single embedding')

        inputs[model.feature] = kwargs.get(model.feature, embeddings[model.feature].create_placeholder(name=model.feature))
        model.sess = kwargs.get('sess', create_session())
        model.pdrop_value = 0.0
        model.dropin_value = {}
        model.embed_model = model.embed(**kwargs)
        model.encoder_model = model.encoder(**kwargs)
        return model

    def encoder(self, **kwargs):
        self.vdrop = kwargs.get('variational', False)
        rnntype = kwargs.get('rnntype', 'blstm')
        nlayers = int(kwargs.get('layers', 1))
        hsz = int(kwargs['hsz'])

        Encoder = BiLSTMEncoderSequence if rnntype == 'blstm' else LSTMEncoderSequence
        return Encoder(None, hsz, nlayers, self.pdrop_value, self.vdrop)

    def encode(self, x=None):
        if x is None:
            x = self.create_placeholder(self.feature)
        self.x = x

        # Calculate the lengths of each word
        lengths = tf.reduce_sum(tf.cast(tf.not_equal(tf.reduce_sum(self.x, axis=2), Offsets.PAD), tf.int32), axis=1)
        embedded = self.embed_model({self.feature: self.x})
        embedded = (embedded, lengths)
        transduced = self.encoder_model(embedded)
        saver = tf.compat.v1.train.Saver()
        saver.restore(self.sess, self.checkpoint)
        return transduced

    def embed(self, **kwargs):
        """This method performs "embedding" of the inputs.  The base method here then concatenates along depth
        dimension to form word embeddings
        :param kwargs:

        :return: A layer representing the embeddings
        """
        return EmbeddingsStack(self.embeddings, self.pdrop_value)

    def get_vsz(self):
        return self.embeddings

    def detached_ref(self):
        return self

