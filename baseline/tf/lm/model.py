"""Language model baselines in TensorFlow
"""
from itertools import chain
from baseline.tf.tfy import *
from baseline.version import __version__
from baseline.model import LanguageModel, register_model
from baseline.tf.embeddings import *
from baseline.tf.tfy import new_placeholder_dict, TRAIN_FLAG, lstm_cell_w_dropout
from baseline.utils import read_json, write_json, MAGIC_VARS


class LanguageModelBase(tf.keras.Model, LanguageModel):
    """Base for all baseline implementations of LMs

    This class provides a loose skeleton around which the baseline models
    are built.  This essentially consists of dividing up the network into a logical separation between "embedding",
    or composition of lookup tables to build a vector representation of a temporal input, "decoding",
    or the conversion of temporal data to a decoded representation, and "output" --
    a projection to output space and a softmax
    """
    def __init__(self):
        """Construct a base LM
        """
        super().__init__()
        self.saver = None
        self.hsz = None
        self.probs = None
        self._unserializable = []

    def save_values(self, basename):
        """Save tensor files out

        :param basename: Base name of model
        :return:
        """
        if not tf.executing_eagerly():
            self.saver.save(self.sess, basename, write_meta_graph=False)
        else:
            self.save_weights(f"{basename}.wgt")

    def save_md(self, basename):
        """This method saves out a `.state` file containing meta-data from these classes and any info
        registered by a user-defined derived class as a `property`. Also write the `graph` and `saver` and `labels`

        :param basename:
        :return:
        """

        write_json(self._state, basename + '.state')
        for key, embedding in self.embeddings.items():
            embedding.save_md(basename + '-{}-md.json'.format(key))

    def _record_state(self, embeddings, **kwargs):
        """
        First, write out the embedding names, so we can recover those.  Then do a deepcopy on the model init params
        so that it can be recreated later.  Anything that is a placeholder directly on this model needs to be removed

        :param kwargs:
        :return:
        """
        embeddings_info = {}
        for k, v in embeddings.items():
            embeddings_info[k] = v.__class__.__name__

        blacklist = set(chain(self._unserializable, MAGIC_VARS, embeddings.keys()))
        self._state = {k: v for k, v in kwargs.items() if k not in blacklist}
        self._state.update({
            'version': __version__,
            'module': self.__class__.__module__,
            'class': self.__class__.__name__,
            'embeddings': embeddings_info,
        })

    def set_saver(self, saver):
        """Connect a `tf.Saver` to the model

        :param saver: A saver
        :return: None
        """
        self.saver = saver

    def save(self, basename):
        """Save the model

        :param basename: The model prefix
        :return:
        """
        self.save_md(basename)
        self.save_values(basename)

    def _create_loss(self, scope):
        with tf.compat.v1.variable_scope(scope):
            targets = tf.reshape(self.y, [-1])
            bt_x_v = tf.nn.log_softmax(tf.reshape(self.logits, [-1, self.vsz]), axis=-1)
            one_hots = tf.one_hot(targets, self.vsz)
            example_loss = -tf.reduce_sum(one_hots * bt_x_v, axis=-1)
            loss = tf.reduce_mean(example_loss)
            return loss

    def create_loss(self):
        """Create training loss operator

        :return: loss
        """
        return self._create_loss(scope='loss')

    def create_test_loss(self):
        """Create test loss operator

        :return: loss
        """
        return self._create_loss(scope='test_loss')

    def make_input(self, batch_dict, train=False):
        """When we are running with `DataFeed`s, need to transform to `feed_dict`s

        :param batch_dict: The batch for a step
        :param train: (`bool`) Are we training (or evaluating)?
        :return: A `feed_dict`
        """
        if not tf.executing_eagerly():
            batch_dict_for_model = new_placeholder_dict(train)

            for key in self.src_keys:
                batch_dict_for_model["{}:0".format(key)] = batch_dict[key]

            y = batch_dict.get('y')
            if y is not None:
                batch_dict_for_model[self.y] = batch_dict['y']

        else:
            SET_TRAIN_FLAG(train)

            batch_dict_for_model = {}
            for key in self.src_keys:
                batch_dict_for_model[key] = batch_dict[key]

        return batch_dict_for_model

    def predict(self, batch_dict):
        """Do prediction from a `batch_dict`

        :param batch_dict: A step of data
        :return: The softmax output for this step
        """
        batch_dict = self.make_input(batch_dict)
        if not tf.executing_eagerly():
            step_softmax = self.sess.run(self.probs, batch_dict)
        else:
            # FIXME: This is not really the proper handling for eager mode
            # We want to be able to pass in the last hidden state and emit the current one right?
            step_softmax = tf.nn.softmax(self(batch_dict, None)[0])

        return step_softmax

    @classmethod
    def create(cls, embeddings, **kwargs):
        """Create the language model

        :param embeddings: A set of embeddings used
        :param kwargs: see below

        :Keyword Arguments:

        * *tgt_key* (`str`) -- Which vocabulary is the destination vocabulary
          (for example, you might have character inputs, or character + word inputs.  The outputs need to be specified)
        * *sess* (`tf.compat.v1.Session`) -- Optionally, pass in a session (or one will be created)
        * *pdrop* (`float`) -- The dropout probability
        * *y* -- Optional target.  If this is not passed in, a placeholder gets created
        * *hsz* (`int`) -- Number of hidden units per layers
        * *unif* (`float`) -- set the weights initializer to small random uniform values

        :return: The created model
        """
        lm = cls()
        lm.src_keys = kwargs.get('src_keys', embeddings.keys())
        lm.tgt_key = kwargs.get('tgt_key')
        if lm.tgt_key is None:
            raise Exception('Need a `tgt_key` to know which source vocabulary should be used for destination')

        lm._unserializable.append(lm.tgt_key)
        lm._record_state(embeddings, **kwargs)
        inputs = {}

        if not tf.executing_eagerly():

            for k, embedding in embeddings.items():
                x = kwargs.get(k, embedding.create_placeholder(name=k))
                inputs[k] = x

            lm.y = kwargs.get('y', tf.compat.v1.placeholder(tf.int32, [None, None], name="y"))
            lm.sess = kwargs.get('sess', tf.compat.v1.Session())
        lm.create_layers(embeddings, **kwargs)

        if not tf.executing_eagerly():
            if lm.requires_state:
                lm.zero_state(inputs)
                lm.logits, lm.final_state = lm(inputs, lm.initial_state)
            else:
                lm.logits, _ = lm(inputs, None)
            lm.probs = tf.nn.softmax(lm.logits, name="softmax")

        return lm

    def call(self, inputs: Dict[str, TensorDef], hidden: TensorDef) -> Tuple[TensorDef, TensorDef]:
        """Take the input and produce the best path of labels out

        :param inputs: The feature indices for the input
        :return: The output and hidden units
        """

    def create_layers(self, embeddings, **kwargs):
        """This method defines the model itself, and must be overloaded by derived classes

        This function will update `self` with the layers required to execute the `call()` method

        :param embeddings: The input feature indices
        :param kwargs:
        :return:
        """

    @classmethod
    def load(cls, basename, **kwargs):
        """Reload the model from a graph file and a checkpoint

        The model that is loaded is independent of the pooling and stacking layers, making this class reusable
        by sub-classes.

        :param basename: The base directory to load from
        :param kwargs: See below

        :Keyword Arguments:
        * *sess* -- An optional tensorflow session.  If not passed, a new session is
            created

        :return: A restored model
        """
        _state = read_json(basename + '.state')

        if not tf.executing_eagerly():
            _state['sess'] = kwargs.pop('sess', create_session())
            embeddings_info = _state.pop("embeddings")

            with _state['sess'].graph.as_default():
                embeddings = reload_embeddings(embeddings_info, basename)
                for k in embeddings_info:
                    if k in kwargs:
                        _state[k] = kwargs[k]

                _state['model_type'] = kwargs.get('model_type', 'default')
                model = cls.create(embeddings, **_state)
                model._state = _state

                do_init = kwargs.get('init', True)
                if do_init:
                    init = tf.compat.v1.global_variables_initializer()
                    model.sess.run(init)

                model.saver = tf.compat.v1.train.Saver()
                model.saver.restore(model.sess, basename)
        else:
            _state = read_json(basename + '.state')
            _state['model_type'] = kwargs.get('model_type', 'default')
            embeddings = {}
            embeddings_dict = _state.pop("embeddings")

            for key, class_name in embeddings_dict.items():
                md = read_json('{}-{}-md.json'.format(basename, key))
                embed_args = dict({'vsz': md['vsz'], 'dsz': md['dsz']})
                Constructor = eval(class_name)
                embeddings[key] = Constructor(key, **embed_args)

            model = cls.create(embeddings, **_state)
            model._state = _state
            model.load_weights(f"{basename}.wgt")

        return model

    @property
    def requires_state(self):
        pass


class AbstractGeneratorModel(LanguageModelBase):

    def create_layers(self, embeddings, **kwargs):
        self.embeddings = self.init_embed(embeddings, **kwargs)
        self.embeddings_proj = self.init_embeddings_proj(**kwargs)
        self.generator = self.init_generate(**kwargs)
        self.output_layer = self.init_output(embeddings, **kwargs)

    def call(self, inputs: Dict[str, TensorDef], hidden: TensorDef) -> Tuple[TensorDef, TensorDef]:
        emb = self.embed(inputs)
        output, hidden = self.generate(emb, hidden)
        return self.output_layer(output), hidden

    def embed(self, input):
        embedded_dropout = self.embeddings(input)
        return self.embeddings_proj(embedded_dropout)

    def init_embed(self, embeddings: Dict[str, TensorDef], **kwargs) -> BaseLayer:
        """This method creates the "embedding" layer of the inputs, with an optional reduction

        :param embeddings: A dictionary of embeddings

        :Keyword Arguments: See below
        * *embeddings_reduction* (defaults to `concat`) An operator to perform on a stack of embeddings
        * *embeddings_dropout = float(kwargs.get('embeddings_dropout', 0.0))

        :return: The output of the embedding stack followed by its reduction.  This will typically be an output
          with an additional dimension which is the hidden representation of the input
        """
        reduction = kwargs.get('embeddings_reduction', 'concat')
        embeddings_dropout = float(kwargs.get('embeddings_dropout', 0.0))
        return EmbeddingsStack({k: embeddings[k] for k in self.src_keys}, embeddings_dropout, reduction=reduction)

    def init_embeddings_proj(self, **kwargs):
        input_sz = self.embeddings.output_dim
        hsz = kwargs.get('hsz', kwargs.get('d_model'))
        if hsz != input_sz:
            proj = tf.keras.layers.Dense(hsz)
            print('Applying a transform from {} to {}'.format(input_sz, hsz))
        else:
            proj = PassThru(hsz)
        return proj

    def init_generate(self, **kwargs):
        pass

    def generate(self, emb, hidden):
        return self.generator((emb, hidden))

    def init_output(self, embeddings, **kwargs):
        self.vsz = embeddings[self.tgt_key].get_vsz()
        do_weight_tying = bool(kwargs.get('tie_weights', False))
        output_bias = kwargs.get('output_bias', False)
        if do_weight_tying:
            output = WeightTieDense(embeddings[self.tgt_key], use_bias=output_bias)
        else:
            output = tf.keras.layers.Dense(self.vsz)
        return output


@register_model(task='lm', name='default')
class RNNLanguageModel(AbstractGeneratorModel):
    """RNN-based Language Model built on base class
    """
    def __init__(self):
        """Construct an RNNLM
        """
        super().__init__()
        self.rnntype = 'lstm'
        self.initial_state = None

    @property
    def requires_state(self):
        return True

    def zero_state(self, inputs):
        batchsz = get_shape_as_list(inputs[self.src_keys[0]])[0]
        self.initial_state = self.generator.layer.zero_state(batchsz)

    def init_generate(self, **kwargs):
        """LSTM-based method for decoding

        :param inputs: The outputs of the embeddings
        :param kwargs: See above

        :return: The layer
        """
        pdrop = float(kwargs.get('dropout', 0.5))
        layers = kwargs.get('layers', kwargs.get('num_layers', 1))
        self.hsz = kwargs.get('hsz', kwargs.get('d_model'))
        return WithDropoutOnFirst(LSTMEncoderWithState(self.hsz, self.hsz, layers, pdrop, batch_first=True),
                                  pdrop,
                                  kwargs.get('variational', False))


@register_model(task='lm', name='transformer')
class TransformerLanguageModel(AbstractGeneratorModel):
    """Transformer-based Language Model built on base class
    """
    def __init__(self):
        """Construct an TLM
        """
        super().__init__()

    @property
    def requires_state(self):
        return False

    def init_generate(self, **kwargs):
        pdrop = float(kwargs.get('dropout', 0.1))
        layers = kwargs.get('layers', kwargs.get('num_layers', 1))
        d_model = int(kwargs.get('d_model', kwargs.get('hsz')))
        num_heads = kwargs.get('num_heads', 4)
        d_ff = int(kwargs.get('d_ff', 4 * d_model))
        rpr_k = kwargs.get('rpr_k')
        d_k = kwargs.get('d_k')
        scale = bool(kwargs.get('scale', True))
        activation = kwargs.get('activation', 'gelu')
        ffn_pdrop = kwargs.get('ffn_pdrop', 0.0)
        layer_norm_eps = kwargs.get('layer_norm_eps', 1e-12)
        layer_norms_after = kwargs.get('layer_norms_after', False)
        layer_drop = kwargs.get('layer_drop', 0.0)
        windowed_ra = kwargs.get('windowed_ra', False)
        rpr_value_on = kwargs.get('rpr_value_on', True)
        return TransformerEncoderStack(num_heads, d_model=d_model, pdrop=pdrop, scale=scale,
                                       layers=layers, d_ff=d_ff, rpr_k=rpr_k, d_k=d_k,
                                       activation=activation,
                                       ffn_pdrop=ffn_pdrop,
                                       layer_norm_eps=layer_norm_eps,
                                       layer_norms_after=layer_norms_after, windowed_ra=windowed_ra,
                                       rpr_value_on=rpr_value_on,
                                       layer_drop=layer_drop)

    def create_mask(self, bth):
        max_seqlen = get_shape_as_list(bth)[1]
        mask = subsequent_mask(max_seqlen)
        return mask

    def generate(self, bth, _):
        mask = self.create_mask(bth)
        return self.generator((bth, mask)), None


@register_model(task='lm', name='transformer-mlm')
class TransformerMaskedLanguageModel(TransformerLanguageModel):

    def create_mask(self, bth):
        return None

