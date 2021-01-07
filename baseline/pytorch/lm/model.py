from baseline.pytorch.torchy import *
from baseline.pytorch.transformer import TransformerEncoderStack, subsequent_mask, MultiHeadedAttention
from baseline.model import LanguageModel, register_model
import torch.autograd
import os


class LanguageModelBase(nn.Module, LanguageModel):
    def __init__(self):
        super().__init__()

    def save(self, outname):
        torch.save(self, outname)
        basename, _ = os.path.splitext(outname)

    def create_loss(self):
        return SequenceCriterion(LossFn=nn.CrossEntropyLoss)

    @classmethod
    def load(cls, filename, **kwargs):
        device = kwargs.get('device')
        if not os.path.exists(filename):
            filename += '.pyt'
        model = torch.load(filename, map_location=device)
        model.gpu = False if device == 'cpu' else model.gpu
        return model

    def zero_state(self, batchsz):
        return None

    @property
    def requires_state(self):
        pass

    def make_input(self, batch_dict, numpy_to_tensor=False):
        example_dict = dict({})
        for key in self.src_keys:

            tensor = batch_dict[key]
            if numpy_to_tensor:
                tensor = torch.from_numpy(tensor)

            if self.gpu:
                tensor = tensor.cuda()

            example_dict[key] = tensor

        y = batch_dict.get('y')
        if y is not None:
            if numpy_to_tensor:
                y = torch.from_numpy(y)
            if self.gpu:
                y = y.cuda()
            example_dict['y'] = y
        return example_dict

    @classmethod
    def create(cls, embeddings, **kwargs):

        lm = cls()
        lm.gpu = kwargs.get('gpu', True)
        lm.tgt_key = kwargs.get('tgt_key')
        if lm.tgt_key is None:
            raise Exception('Need a `tgt_key` to know which source vocabulary should be used for destination ')

        lm.src_keys = kwargs.get('src_keys', embeddings.keys())
        lm.create_layers(embeddings, **kwargs)
        return lm

    def create_layers(self, embeddings, **kwargs):
        """This method defines the model itself, and must be overloaded by derived classes

        This function will update `self` with the layers required to execute the `call()` method

        :param embeddings: The input feature indices
        :param kwargs:
        :return:
        """

    def predict(self, batch_dict, **kwargs):
        self.eval()
        numpy_to_tensor = bool(kwargs.get('numpy_to_tensor', True))
        batch_dict = self.make_input(batch_dict, numpy_to_tensor=numpy_to_tensor)
        hidden = batch_dict.get('h')
        step_softmax, _ = self(batch_dict, hidden)
        return F.softmax(step_softmax, dim=-1)


class AbstractGeneratorLanguageModel(LanguageModelBase):

    def create_layers(self, embeddings, **kwargs):
        self.embeddings = self.init_embed(embeddings, **kwargs)
        self.embeddings_proj = self.init_embeddings_proj(**kwargs)
        self.generator = self.init_generate(**kwargs)
        self.output_layer = self.init_output(embeddings, **kwargs)

    def forward(self, input: Dict[str, TensorDef], hidden: TensorDef) -> Tuple[TensorDef, TensorDef]:
        emb = self.embed(input)
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
            proj = pytorch_linear(input_sz, hsz)
            print('Applying a transform from {} to {}'.format(input_sz, hsz))
        else:
            proj = nn.Identity()
        return proj

    def init_generate(self, **kwargs):
        pass

    def generate(self, emb, hidden):
        return self.generator((emb, hidden))

    def init_output(self, embeddings, **kwargs):
        self.vsz = embeddings[self.tgt_key].get_vsz()
        hsz = kwargs.get('hsz', kwargs.get('d_model'))
        unif = float(kwargs.get('unif', 0.0))
        do_weight_tying = bool(kwargs.get('tie_weights', False))
        output_bias = kwargs.get('output_bias', False)
        if do_weight_tying:
            output = WeightTieDense(embeddings[self.tgt_key], output_bias)
        else:
            output = pytorch_linear(hsz, self.vsz, unif)
        return output


@register_model(task='lm', name='default')
class RNNLanguageModel(AbstractGeneratorLanguageModel):

    def __init__(self):
        super().__init__()

    def zero_state(self, batchsz):
        weight = next(self.parameters()).data
        return (torch.autograd.Variable(weight.new(self.num_layers, batchsz, self.hsz).zero_()),
                torch.autograd.Variable(weight.new(self.num_layers, batchsz, self.hsz).zero_()))

    @property
    def requires_state(self):
        True

    def init_generate(self, **kwargs):
        pdrop = float(kwargs.get('dropout', 0.5))
        self.num_layers = kwargs.get('layers', kwargs.get('num_layers', 1))
        self.hsz = kwargs.get('hsz', kwargs.get('d_model'))
        return WithDropoutOnFirst(LSTMEncoderWithState(self.hsz, self.hsz, self.num_layers, pdrop, batch_first=True),
                                  pdrop,
                                  kwargs.get('variational', False))


@register_model(task='lm', name='transformer')
class TransformerLanguageModel(AbstractGeneratorLanguageModel):

    def __init__(self):
        super().__init__()

    @property
    def requires_state(self):
        False

    def init_layer_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding, nn.LayerNorm)):
            module.weight.data.normal_(mean=0.0, std=self.weight_std)
        if isinstance(module, (nn.Linear, nn.LayerNorm)) and module.bias is not None:
            module.bias.data.zero_()

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

    def create_layers(self, embeddings, **kwargs):
        super().create_layers(embeddings, **kwargs)
        self.weight_std = kwargs.get('weight_std', 0.02)
        self.apply(self.init_layer_weights)

    def create_mask(self, bth):
        T = bth.shape[1]
        mask = subsequent_mask(T).type_as(bth)
        return mask

    def generate(self, bth, _):
        mask = self.create_mask(bth)
        return self.generator((bth, mask)), None


@register_model(task='lm', name='transformer-mlm')
class TransformerMaskedLanguageModel(TransformerLanguageModel):

    def create_mask(self, bth):
        return None
