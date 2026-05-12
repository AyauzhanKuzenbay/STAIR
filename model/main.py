"""
STAIR – entry point.

Usage:
    python main.py --config arg.ini
"""

import argparse
from arg_processor import ArgProcessor
from dnn_model import STAIRModel
from model_processor import ModelProcessor
from keras_data_generators import TrainGenerator


def main():
    parser = argparse.ArgumentParser(description="STAIR trajectory retrieval")
    parser.add_argument("--config", required=True,
                        help="Path to the .ini configuration file")
    args = parser.parse_args()

    arg_processor = ArgProcessor(args.config)

    print("Reading training data...")
    train_gen = TrainGenerator(
        arg_processor.training_x_path,
        arg_processor.training_y_path,
        arg_processor.batch_size,
    )
    val_gen = TrainGenerator(
        arg_processor.validation_x_path,
        arg_processor.validation_y_path,
        arg_processor.batch_size,
    )

    print("Building STAIR model...")
    stair_model = STAIRModel(
        embed_vocab_size  = arg_processor.embedding_vocab_size,
        embedding_size    = arg_processor.embedding_size,
        traj_repr_size    = arg_processor.traj_repr_size,
        gru_cell_size     = arg_processor.gru_cell_size,
        num_gru_layers    = arg_processor.num_gru_layers,
        gru_dropout_ratio = arg_processor.gru_dropout_ratio,
        bidirectional     = arg_processor.bidirectional,
        use_attention     = arg_processor.use_attention,
        k                 = max(arg_processor.ks),
    )

    model_processor = ModelProcessor()

    if arg_processor.is_training:
        model_processor.model_train(
            stair_model.model,
            train_gen,
            val_gen,
            arg_processor,
        )

    if arg_processor.is_evaluating:
        model_processor.model_evaluate(
            stair_model.encoder.model,
            arg_processor,
        )


if __name__ == "__main__":
    main()
