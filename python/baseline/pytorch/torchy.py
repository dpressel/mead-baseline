import torch
import numpy as np
from baseline.utils import lookup_sentence, get_version
import torch.autograd
import torch.nn as nn


PYT_MAJOR_VERSION = get_version(torch)


def classify_bt(model, batch_time):
    tensor = torch.from_numpy(batch_time) if type(batch_time) == np.ndarray else batch_time
    probs = model(torch.autograd.Variable(tensor, requires_grad=False).cuda()).exp().data
    probs.div_(torch.sum(probs))
    results = []
    batchsz = probs.size(0)
    for b in range(batchsz):
        outcomes = [(model.labels[id_i], prob_i) for id_i, prob_i in enumerate(probs[b])]
        results.append(outcomes)
    return results


def predict_seq_bt(model, x, xch, lengths):
    x_t = torch.from_numpy(x) if type(x) == np.ndarray else x
    xch_t = torch.from_numpy(xch) if type(xch) == np.ndarray else xch
    len_v = torch.from_numpy(lengths) if type(lengths) == np.ndarray else lengths
    x_v = torch.autograd.Variable(x_t, requires_grad=False).cuda()
    xch_v = torch.autograd.Variable(xch_t, requires_grad=False).cuda()
    #len_v = torch.autograd.Variable(len_t, requires_grad=False)
    results = model((x_v, xch_v, len_v))
    #print(results)
    #if type(x) == np.ndarray:
    #    # results = results.cpu().numpy()
    #    # Fix this to not be greedy
    #    results = np.argmax(results, -1)

    return results


def to_scalar(var):
    # returns a python float
    return var.view(-1).data.tolist()[0]


def argmax(vec):
    # return the argmax as a python int
    _, idx = torch.max(vec, 1)
    return to_scalar(idx)


def log_sum_exp(vec):
    max_score = vec[0, argmax(vec)]
    max_score_broadcast = max_score.view(1, -1).expand(1, vec.size()[1])
    return max_score + torch.log(torch.sum(torch.exp(vec - max_score_broadcast)))


class SequenceCriterion(nn.Module):

    def __init__(self, LossFn=nn.NLLLoss):
        super(SequenceCriterion, self).__init__()
        self.crit = LossFn(ignore_index=0, size_average=False)

    def forward(self, inputs, targets):
        # This is BxT, which is what we want!
        total_sz = targets.nelement()
        loss = self.crit(inputs.view(total_sz, -1), targets.view(total_sz))
        return loss


class StackedLSTMCell(nn.Module):
    def __init__(self, num_layers, input_size, rnn_size, dropout):
        super(StackedLSTMCell, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.num_layers = num_layers
        self.layers = nn.ModuleList()

        for i in range(num_layers):
            self.layers.append(nn.LSTMCell(input_size=input_size, hidden_size=rnn_size, bias=False))
            input_size = rnn_size

    def forward(self, input, hidden):
        h_0, c_0 = hidden
        hs, cs = [], []
        for i, layer in enumerate(self.layers):
            h_i, c_i = layer(input, (h_0[i], c_0[i]))
            input = h_i
            if i != self.num_layers - 1:
                input = self.dropout(input)
            hs += [h_i]
            cs += [c_i]

        hs = torch.stack(hs)
        cs = torch.stack(cs)

        return input, (hs, cs)


class StackedGRUCell(nn.Module):
    def __init__(self, num_layers, input_size, rnn_size, dropout):
        super(StackedGRUCell, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.num_layers = num_layers
        self.layers = nn.ModuleList()

        for i in range(num_layers):
            self.layers.append(nn.GRUCell(input_size=input_size, hidden_size=rnn_size))
            input_size = rnn_size

    def forward(self, input, hidden):
        h_0 = hidden
        hs = []
        for i, layer in enumerate(self.layers):
            h_i = layer(input, (h_0[i]))
            input = h_i
            if i != self.num_layers:
                input = self.dropout(input)
            hs += [h_i]

        hs = torch.stack(hs)

        return input, hs


def pytorch_rnn_cell(insz, hsz, rnntype, nlayers, dropout):

    if rnntype == 'gru':
        rnn = StackedGRUCell(nlayers, insz, hsz, dropout)
    else:
        rnn = StackedLSTMCell(nlayers, insz, hsz, dropout)
    return rnn


def pytorch_embedding(x2vec, finetune=True):
    dsz = x2vec.dsz
    lut = nn.Embedding(x2vec.vsz + 1, dsz, padding_idx=0)
    del lut.weight
    lut.weight = nn.Parameter(torch.FloatTensor(x2vec.weights),
                              requires_grad=finetune)
    return lut


def pytorch_activation(name="relu"):
    if name == "tanh":
        return nn.Tanh()
    if name == "prelu":
        return nn.PReLU()
    if name == "sigmoid":
        return nn.Sigmoid()
    if name == "log_sigmoid":
        return nn.LogSigmoid()
    return nn.ReLU()


def pytorch_conv1d(in_channels, out_channels, fsz, unif=0, padding=0):
    c = nn.Conv1d(in_channels, out_channels, fsz, padding=padding)
    if unif > 0:
        c.weight.data.uniform_(-unif, unif)
    else:
        torch.nn.init.orthogonal(c.weight)
    return c


def pytorch_linear(in_sz, out_sz, unif=0):
    l = nn.Linear(in_sz, out_sz)
    if unif > 0:
        l.weight.data.uniform_(-unif, unif)
    else:
        torch.nn.init.orthogonal(l.weight)
    l.bias.data.zero_()
    return l


def pytorch_rnn(insz, hsz, rnntype, nlayers, dropout):

    if rnntype == 'gru':
        rnn = torch.nn.GRU(insz, hsz, nlayers, dropout=dropout)
    else:
        rnn = torch.nn.LSTM(insz, hsz, nlayers, dropout=dropout)

    return rnn


class Highway(nn.Module):

    def __init__(self,
                 input_size):
        super(Highway, self).__init__()
        self.proj = nn.Linear(input_size, input_size)
        self.transform = nn.Linear(input_size, input_size)
        self.transform.bias.data.fill_(-2.0)

    def forward(self, input):
        proj_result = nn.functional.relu(self.proj(input))
        proj_gate = nn.functional.sigmoid(self.transform(input))
        gated = (proj_gate * proj_result) + ((1 - proj_gate) * input)
        return gated


def pytorch_lstm(insz, hsz, rnntype, nlayers, dropout, unif=0, batch_first=False):
    ndir = 2 if rnntype.startswith('b') else 1
    #print('ndir: %d, rnntype: %s, nlayers: %d, dropout: %.2f, unif: %.2f' % (ndir, rnntype, nlayers, dropout, unif))
    rnn = torch.nn.LSTM(insz, hsz, nlayers, dropout=dropout, bidirectional=True if ndir > 1 else False, batch_first=batch_first)#, bias=False)
    if unif > 0:
        for weight in rnn.parameters():
            weight.data.uniform_(-unif, unif)
    else:
        nn.init.orthogonal(rnn.weight_hh_l0)
        nn.init.orthogonal(rnn.weight_ih_l0)

    return rnn, ndir*hsz


def pytorch_prepare_optimizer(model, **kwargs):

    mom = kwargs.get('mom', 0.9)
    optim = kwargs.get('optim', 'sgd')
    eta = kwargs.get('eta', kwargs.get('lr', 0.01))
    decay_rate = float(kwargs.get('decay_rate', 0.0))
    decay_type = kwargs.get('decay_type', None)

    if optim == 'adadelta':
        optimizer = torch.optim.Adadelta(model.parameters(), lr=eta)
    elif optim == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=eta)
    elif optim == 'rmsprop':
        optimizer = torch.optim.RMSprop(model.parameters(), lr=eta)
    elif optim == 'asgd':
        optimizer = torch.optim.ASGD(model.parameters(), lr=eta)
    else:
        optimizer = torch.optim.SGD(model.parameters(), lr=eta, momentum=mom)

    scheduler = None
    if decay_rate > 0.0 and decay_type is not None:
        if decay_type == 'invtime':
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1, gamma=decay_rate)

    return optimizer, scheduler


def append2seq(seq, modules):

    for i, module in enumerate(modules):
        seq.add_module('%s-%d' % (str(module), i), module)


def tensor_max(tensor):
    return tensor.max()


def tensor_shape(tensor):
    return tensor.size()


def tensor_reverse_2nd(tensor):
    idx = torch.LongTensor([i for i in range(tensor.size(1)-1, -1, -1)])
    return tensor.index_select(1, idx)


def long_0_tensor_alloc(dims, dtype=None):
    lt = long_tensor_alloc(dims)
    lt.zero_()
    return lt


def long_tensor_alloc(dims, dtype=None):
    if type(dims) == int or len(dims) == 1:
        return torch.LongTensor(dims)
    return torch.LongTensor(*dims)


# Mashed together from code using numpy only, hacked for th Tensors
# This function should never be used for decoding.  It exists only so that the training model can greedily decode
# It is super slow and doesnt use maintain a beam of hypotheses
def show_examples_pytorch(model, es, rlut1, rlut2, embed2, mxlen, sample, prob_clip, max_examples, reverse):
    si = np.random.randint(0, len(es))

    batch_dict = es[si]

    src_array = batch_dict['src']
    tgt_array = batch_dict['dst']
    src_len = batch_dict['src_len']
    #src_array, tgt_array, src_len, _ = es[si]

    if max_examples > 0:
        max_examples = min(max_examples, src_array.size(0))
        src_array = src_array[0:max_examples]
        tgt_array = tgt_array[0:max_examples]
        src_len = src_len[0:max_examples]

    GO = embed2.vocab['<GO>']
    EOS = embed2.vocab['<EOS>']

    # TODO: fix this, check for GPU first
    src_array = src_array.cuda()
    
    for src_len,src_i,tgt_i in zip(src_len, src_array, tgt_array):

        print('========================================================================')

        sent = lookup_sentence(rlut1, src_i.cpu().numpy(), reverse=reverse)
        print('[OP] %s' % sent)
        sent = lookup_sentence(rlut2, tgt_i)
        print('[Actual] %s' % sent)
        dst_i = torch.zeros(1, mxlen).long()
        #if use_gpu:
        dst_i = dst_i.cuda()

        next_value = GO
        src_i = src_i.view(1, -1)
        for j in range(mxlen):
            dst_i[0,j] = next_value
            probv = model((torch.autograd.Variable(src_i), torch.autograd.Variable(dst_i)))
            output = probv.squeeze()[j]
            if sample is False:
                _, next_value = torch.max(output, 0)
                next_value = int(next_value.data[0])
            else:
                probs = output.data.exp()
                # This is going to zero out low prob. events so they are not
                # sampled from
                best, ids = probs.topk(prob_clip, 0, largest=True, sorted=True)
                probs.zero_()
                probs.index_copy_(0, ids, best)
                probs.div_(torch.sum(probs))
                fv = torch.multinomial(probs, 1)[0]
                next_value = fv

            if next_value == EOS:
                break

        sent = lookup_sentence(rlut2, dst_i.squeeze())
        print('Guess: %s' % sent)
        print('------------------------------------------------------------------------')
