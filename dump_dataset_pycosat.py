import tensorflow as tf
import numpy as np

import cnf_dataset
from tqdm import tqdm
import argparse
import os
import random


def make_example(inputs, sat):
    example = tf.train.Example(
        features=tf.train.Features(feature={
            "inputs": tf.train.Feature(float_list=tf.train.FloatList(value=list(inputs.flatten()))),
            "sat": tf.train.Feature(
                float_list=tf.train.FloatList(value=list(sat.flatten())))
        })
    )
    return example.SerializeToString()


def tf_serialize_example(sample):
    tf_string = tf.py_func(make_example, (sample["inputs"], sample["sat"]), tf.string)
    return tf.reshape(tf_string, ())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-j", "--job", required=True, help="the job identifier")
    ap.add_argument("-c", "--complexity", required=True, type=int, default=30, help="the level of complexity of SR(n)")
    ap.add_argument("-o", "--observations", required=True, type=int, default=1e4, help="the number of observations to be made")
    args = vars(ap.parse_args())

    random.seed(int(job))
    print("Set random seed to {}".format(int(job)))

    dirname = "sr_{}".format(complexity)  
    filename = "train_{}_sr_{}.tfrecord".format(job, complexity)
    options = {
        "PROCESSOR_NUM": 24,
        "CLAUSE_NUM": 2*complexity,
        "VARIABLE_NUM": complexity,
        "MIN_VARIABLE_NUM": 1,
        "BATCH_SIZE": 1,
        "CLAUSE_SIZE": 2,
        "MIN_CLAUSE_NUM": 2,
        "SR_GENERATOR": False
    }
    #n_observations = args["observations"]

    if not os.path.exists(dirname):
        os.makedirs(dirname)
        print("Created directory {}".format(dirname))

    with cnf_dataset.PoolDatasetGenerator(options) as generator, \
            tf.python_io.TFRecordWriter(os.path.join(dirname, filename)) as writer:

        for i in range(1, complexity):
            sample_with_labels = generator.generate_batch(representation='cnfs')

            alfa = 2 * i
            cnfs = tcnfgen(alfa, complexity)
            sat_labels = np.array([(pysat.solve(cnfs)!='UNSAT')])
            inputs = np.array([cnfs])

            print(sat_labels)
            print(inputs)

            tf_sample = {
                 "inputs": np.squeeze(inputs.astype(np.float32), 0),
                  "sat": np.squeeze(np.asarray(sat_labels).astype(np.float32), 0)
            }

            serialized = make_example(**tf_sample)
            writer.write(serialized)
