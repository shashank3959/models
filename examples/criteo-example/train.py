import os
import argparse
import pandas as pd

import nvtabular as nvt
from merlin.models.utils.example_utils import workflow_fit_transform
import merlin.io
import merlin.models.tf as mm

from nvtabular import ops
from merlin.core.utils import download_file
from merlin.datasets.entertainment import get_movielens
from merlin.schema.tags import Tags

import tensorflow as tf
import keras

from time import time


VERBOSE=False # print extra information about model structure


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--data_dir', type=str, default='/mm-benchmark-criteo/criteo-preprocessed-data')
    parser.add_argument('--embedding_dim', type=int, default=32)
    parser.add_argument('--part_mem_fraction', type=float, default=0.08)

    # Training time args
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--warmup_epochs', type=int, default=0)
    parser.add_argument('--batch_size', type=int, default=32768)
    parser.add_argument('--use_sgd', action='store_true', help='Use SGD optimizer instead of Ftrl/Adagrad')
    parser.add_argument('--lr', type=float, default=0.08)
    parser.add_argument('--no_gpu', action='store_true', help='Use CPU only', default=False)
    parser.add_argument('--model_save_dir', type=str, default='/mm-benchmark-criteo/model_save')
    parser.add_argument('--model_save_prefix', type=str, default='model-1')

    # Logging
    parser.add_argument('--log_tensorboard', action='store_true', help='Log Tensorboard')
    parser.add_argument('--tensorboard_log_dir', type=str, default='/mm-benchmark-criteo/benchmark-logs')
    parser.add_argument('-v', '--verbose', action='store_true', default=True)
    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()

    if args.verbose:
        VERBOSE=True
        print("="*20)
        print(args)


    train = merlin.io.Dataset(
        os.path.join(args.data_dir, "train"), engine="parquet", part_mem_fraction=args.part_mem_fraction
    )
    valid = merlin.io.Dataset(
        os.path.join(args.data_dir, "valid"), engine="parquet", part_mem_fraction=args.part_mem_fraction
    )


    # MODEL DEFINITION
    model = mm.DLRMModel(
        train.schema,                                                            # 1
        embedding_dim=args.embedding_dim,
        bottom_block=mm.MLPBlock([256, args.embedding_dim]),                         # 2
        top_block=mm.MLPBlock([256, 128, 64]),
        prediction_tasks=mm.BinaryClassificationTask(                            # 3
            train.schema.select_by_tag(Tags.TARGET).column_names[0]
        )               
    )


    if args.use_sgd:
        optimizer = tf.keras.optimizers.SGD(learning_rate=args.lr)
    else:
        optimizer = tf.keras.optimizers.Adam(learning_rate=args.lr)

    # Compile
    model.compile(optimizer=optimizer)

    # Warmup
    if args.warmup_epochs > 0:
        model.fit(train, batch_size=args.batch_size)

    # Train
    train_start = time()
    model.fit(train, batch_size=args.batch_size)
    print("=== TOTAL TRAINING TIME === :", time() - train_start)


    # Validation
    valid_start = time()
    metrics = model.evaluate(valid, batch_size=args.batch_size, return_dict=True)
    print("=== TOTAL VALIDATION TIME === :", time() - valid_start)
    
    if VERBOSE:
        print(model.summary())

    print("VALIDATION METRICS:")
    print(metrics)

    model_save_path = os.path.join(args.model_save_dir, args.model_save_prefix)
    # Save Model
    model = keras.models.load_model(model_save_path)
    if VERBOSE:
        print("Model saved at:", model_save_path)
