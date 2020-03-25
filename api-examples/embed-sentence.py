import argparse
import baseline
import sys
sys.path.append('../python/addons')
from baseline.utils import import_user_module
from baseline.tf.embeddings import *
from baseline.embeddings import *
from baseline.vectorizers import *
import tensorflow as tf
import numpy as np

if get_version(tf) >= 2:
    tf.compat.v1.disable_eager_execution()


def get_pool_op(s):
    """Allows us to pool the features with either ``max`` or ``mean``. O.w. use identity

    :param s: The operator
    :return: The pool operation
    """
    return np.mean if s == 'mean' else np.max if s == 'max' else lambda x, axis: x


def get_vectorizer(vec_type, vf, mxlen, mxwlen, lower=True):
    """Get a vectorizer object by name from `BASELINE_VECTORIZERS` registry

    :param vf: A vocabulary file (which might be ``None``)
    :param mxlen: The vector length to use
    :param lower: (``bool``) should we lower case?  Defaults to ``True``
    :return: A ``baseline.Vectorizer`` subclass
    """
    transform_fn = baseline.lowercase if lower else None
    return create_vectorizer(type=vec_type, transform_fn=transform_fn, vocab_file=vf, mxlen=mxlen, mxwlen=mxwlen)

def get_embedder(embed_type, embed_file, sess, feature='word'):
    """Get an embedding object by type so we can evaluate one hot vectors

    :param embed_type: (``str``) The name of the embedding in the `BASELINE_EMBEDDINGS`
    :param embed_file: (``str``) Either the file or a URL to a hub location for the model
    :return: An embeddings dict containing vocab and graph
    """
    if embed_type == 'bert':
        embed_type += '-embed'
    # We have to create a key for each embedding we make.  Here we just call it 'word'
    embed = baseline.load_embeddings(feature, embed_type=embed_type,
                                     embed_file=embed_file, keep_unused=True, trainable=False, sess=sess)
    return embed


parser = argparse.ArgumentParser(description='Encode a sentence as an embedding')
parser.add_argument('--embed_file', help='embedding file')
parser.add_argument('--type', default='default')
parser.add_argument('--sentences', required=True)
parser.add_argument('--output', default=None)
parser.add_argument('--pool', default=None)
parser.add_argument('--lower', type=baseline.str2bool, default=True)
parser.add_argument('--vocab_file')
parser.add_argument('--max_length', type=int, default=100)
parser.add_argument('--max_word_length', type=int, default=40)
parser.add_argument('--vec_type', default='token1d', type=str)
parser.add_argument('--modules', help='tasks to load, must be local', default=[], nargs='+', required=False)
parser.add_argument('--feature', help='feature to use', default='word')
args = parser.parse_args()

# task_module overrides are not allowed via hub or HTTP, must be defined locally
for module in args.modules:
    import_user_module(module)
# Create our vectorizer according to CL
vectorizer = get_vectorizer(args.vec_type, args.vocab_file, args.max_length, args.max_word_length, args.lower)

# Pool operation for once we have np.array
pooling = get_pool_op(args.pool)

# Make a session
with tf.compat.v1.Session() as sess:
    # Get embeddings
    embed = get_embedder(args.type, args.embed_file, sess=sess, feature=args.feature)

    # This is our decoder graph object
    embedder = embed['embeddings']

    # This is the vocab
    vocab = embed['vocab']

    # Declare a tf graph operation
    y = embedder(None)
    sess.run(tf.compat.v1.global_variables_initializer())

    # Read a newline separated file of sentences
    with open(args.sentences) as f:

        vecs = []
        for line in f:
            # For each sentence
            tokens = line.strip().split()
            # Run vectorizer to get ints and length of vector
            one_hot, sentence_len = vectorizer.run(tokens, vocab)
            # Expand so we have a batch dim
            one_hot_batch = np.expand_dims(one_hot, 0)
            # Evaluate the graph and get rid of batch dim
            sentence_emb = sess.run(y, feed_dict={embedder.x: one_hot_batch})#.squeeze()
            # Run the pooling operator
            sentence_vec = pooling(sentence_emb, axis=0)
            vecs.append(sentence_vec)
        # Store to file
        if args.output:
            np.savez(args.output, np.stack(vecs))
        else:
            print(vecs)
