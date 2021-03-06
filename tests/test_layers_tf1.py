import pytest
import numpy as np
from eight_mile.utils import get_version
tf = pytest.importorskip('tensorflow')
pytestmark = pytest.mark.skipif(get_version(tf) >= 2, reason="TF2.X")
from eight_mile.utils import Offsets
from eight_mile.tf.layers import infer_lengths

B = 10
T = 15
TRIALS = 100

@pytest.fixture
def lengths():
    lengths = np.random.randint(1, T, size=(B,)).astype(np.int32)
    return lengths


def generate_data_with_zeros(lengths):
    data = np.random.randint(1, 100, (len(lengths), np.max(lengths))).astype(np.int32)
    for i, length in enumerate(lengths):
        data[i, length:] = 0
        if length // 2 > 0:
            extra_zeros = np.random.randint(0, length.item() - 1, size=((length // 2).item(),))
            data[i, extra_zeros] = 0
    return data


def test_infer_lengths(lengths):
    def test():
        tf.compat.v1.reset_default_graph()
        with tf.compat.v1.Session() as sess:
            data = generate_data_with_zeros(lengths)
            data = tf.constant(data)
            infered = infer_lengths(data, axis=1)
            infered = sess.run(infered)
        np.testing.assert_allclose(infered, lengths)

    for _ in range(TRIALS):
        test()


def test_infer_lengths_t_first(lengths):
    def test():
        tf.compat.v1.reset_default_graph()
        with tf.compat.v1.Session() as sess:
            data = generate_data_with_zeros(lengths)
            print(data)
            data = tf.constant(data)
            data = tf.transpose(data)
            print(data)
            infered = infer_lengths(data, axis=0)
            infered = sess.run(infered)
            print(infered)
        np.testing.assert_allclose(infered, lengths)

    for _ in range(TRIALS):
        test()


def test_infer_lengths_multi_dim():
    data = tf.random.uniform((10, 11, 12))
    with pytest.raises(ValueError):
        infer_lengths(data, axis=1)
