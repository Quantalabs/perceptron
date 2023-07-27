# Perceptron
A basic customizable single layer perceptron.

## Docs
`python train.py --train` - Trains the model \
`python train.py --test` - Test the model on the dataset \
`python train.py --gradio` - Run a gradio instance \
`python train.py --help` - Help message \
`python train.py --testOnFlagged` - Test the model on the images flagged by gradio \
`python train.py --trainOnFlagged` - Train the model on the images flagged by gradio \

### Options
`--shape` - The shape of the image. Defaults to 28 \
`--bias` - Defaults to 0 \
`--weights` - `.csv` file containing weights. Defaults to None (will use 0s for all weights initially) \
`--circles` - (for `--train` and `--test` only) Directory for circles to train/test on. Defaults to `data/circles` \
`--squares` - (for `--train` and `--test` only) Directory for squares to train/test on. Defaults to `data/squares` \
`--output` - (for`--train` and `--trainOnFlagged` only) File to save weights to. Will not save weights otherwise. \
`--count` - (for `--train` and `--trainOnFlagged` only) How many epochs to run the model on the dataset. Defaults to 1 \
`--host` - (for `--gradio` only) Host to run the gradio instance on. Defaults to 127.0.0.1 \
`--port` - (for `--gradio` only) Port to run the gradio instance on. Defaults to 8080 \
`--retest` - (for `--trainOnFlagged` only) Will test the model against the flagged images after training \