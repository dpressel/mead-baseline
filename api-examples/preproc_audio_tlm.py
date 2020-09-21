import argparse
import baseline
from baseline.vectorizers import BPEVectorizer1D
from eight_mile.utils import write_yaml, mlm_masking
import json
import struct
import logging
import numpy as np
import os
import soundfile as sf
logger = logging.getLogger('baseline')
try:
    import tensorflow as tf
except:
    pass

"""Read in the WAV files and convert them to a curriculum of buckets

We are going to read the manifest file, create a set of buckets, create masks for the each sample
and read and pad the WAV files.  We also need to generate a full manifest

"""

def create_record(chunk, length):
    """Emit a record

    :param chunk: A chunk of float inputs
    :param An object with `{'x_f': inputs, 'length': length}`
    """

    inputs = np.array(chunk)
    length = np.array([length])
    return {'x_f': inputs, 'length': length}


def in_bytes(mb):
    return mb * 1024 * 1024


def process_sample(file, max_sample_length):
    """Read in a line and turn it into an entry

    The entries will get collated by the data loader

    :param file:
    :return:
    """
    wav, _ = sf.read(file)
    total_sz = len(wav)
    end = total_sz
    start = 0
    if total_sz > max_sample_length:
        diff = total_sz - max_sample_length
        start = np.random.randint(0, diff + 1)
        end = total_sz - diff + start
    v = np.zeros(max_sample_length)
    wav = wav[start:end]
    valid_len = len(wav)
    v[0:valid_len] = wav
    return v, valid_len


class RollingWriter:
    def __init__(self, name, fields, max_file_size_mb):
        self.name = name
        self.counter = 1
        self.fields = fields
        self.current_file_size = 0
        self.writer = None
        self.max_file_size = in_bytes(max_file_size_mb)
        self._rollover_file()

    def _open_file(self, filename):
        return open(filename, 'w')

    def _rollover_file(self):
        if self.writer:
            self.writer.close()
        filename = f'{self.name}-{self.counter}.{self.suffix}'
        self.counter += 1
        self.current_file_size = 0
        logger.info("Rolling over.  New file [%s]", filename)
        self.writer = self._open_file(filename)

    @property
    def suffix(self):
        raise Exception("Dont know suffix in ABC")

    def _write_line(self, str_val):
        self.writer.write(str_val)
        return len(str_val.encode("utf8"))

    def _write_line_rollover(self, l):
        sz = self._write_line(l)
        self.current_file_size += sz
        if self.current_file_size > self.max_file_size:
            self._rollover_file()

    def close(self):
        self.writer.close()


class TFRecordRollingWriter(RollingWriter):
    def __init__(self, name, fields, max_file_size_mb):
        try:
            self.RecordWriterClass = tf.io.TFRecordWriter
        except Exception as e:
            raise Exception("tfrecord package could not be loaded, pip install that first, along with crc32c")
        super().__init__(name, fields, max_file_size_mb)

    @staticmethod
    def _bytes_feature(value):
        """Returns a bytes_list from a string / byte."""
        if isinstance(value, type(tf.constant(0))):
            value = value.numpy()  # BytesList won't unpack a string from an EagerTensor.
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))

    @staticmethod
    def _int64_feature(value):
        """Returns an int64_list from a bool / enum / int / uint."""
        return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

    @staticmethod
    def _float32_feature(value):
        """Returns an int64_list from a bool / enum / int / uint."""
        return tf.train.Feature(float_list=tf.train.FloatList(value=value))

    def _open_file(self, filename):
        return self.RecordWriterClass(filename)

    def _write_line(self, str_val):
        self.writer.write(str_val)
        return len(str_val)

    def serialize_tf_example(self, record):
        """
        Creates a tf.Example message ready to be written to a file.
        """
        # Create a dictionary mapping the feature name to the tf.Example-compatible
        # data type.
        feature = {}
        for f in self.fields:
            if f.endswith('_str'):
                value = ' '.join(record[f])
                value = TFRecordRollingWriter._bytes_feature(value.encode('utf-8'))
            elif f.endswith('_f'):
                value = TFRecordRollingWriter._float32_feature(record[f])
            else:
                value = TFRecordRollingWriter._int64_feature(record[f])
            feature[f] = value

        example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
        return example_proto.SerializeToString()

    def write(self, record):
        example_str = self.serialize_tf_example(record)
        self._write_line_rollover(example_str)

    @property
    def suffix(self):
        return 'tfrecord'

def read_manifest(manifest, bucket_lengths, min_length=None):
    skipped = 0
    asc = sorted(bucket_lengths)
    buckets = {b: [] for b in asc}
    num_samples = 0
    with open(manifest, "r") as f:

        directory = f.readline().strip()
        for line in f:
            num_samples += 1
            items = line.strip().split("\t")
            sz = int(items[1])
            fname = os.path.join(directory, items[0])

            if sz >= asc[-1]:
                buckets[asc[-1]].append((fname, sz))
            else:
                for b in asc:
                    if sz <= b:
                        buckets[b].append((fname, sz))
                        break

            if min_length is not None and sz < min_length:
                skipped += 1
                continue
    logger.info('Num samples %d', num_samples)
    return buckets

parser = argparse.ArgumentParser(description='Convert text into fixed width buckets')

parser.add_argument('--input_files',
                    help='The text to classify as a string, or a path to a file with each line as an example', type=str)
parser.add_argument("--manifest_dir", required=True)
parser.add_argument("--manifest_file", type=str, default="train.tsv", help='File path to use for train file')
parser.add_argument("--output", type=str, required=True, help="Output base name, e.g. /path/to/output/record")
parser.add_argument("--max_file_size", type=int, default=100, help="Shard size, defaults to 100MB")
parser.add_argument("--buckets", type=int, nargs="+", default=[62500, 125000, 250000])
args = parser.parse_args()


manifest = os.path.join(args.manifest_dir, args.manifest_file)
buckets = read_manifest(manifest, bucket_lengths=args.buckets)
logger.info('Bucket distribution: [%s]', ' '.join([str(len(v)) for k, v in buckets.items()]))

root_dir = os.path.dirname(args.output)
if not os.path.exists(root_dir):
    os.makedirs(root_dir)


fw = TFRecordRollingWriter(args.output, ('x_f', 'length'), args.max_file_size)
num_samples = 0

for b in args.buckets:
    for file, sz in buckets[b]:
        v, vsz = process_sample(file, b)
        if sz < vsz:
            print(vsz, sz)
        r = create_record(v, vsz)
        fw.write(r)

write_yaml({'num_samples': num_samples}, os.path.join(root_dir, 'md.yml'))
