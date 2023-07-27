# Perceptron
A basic customizable single layer perceptron.

## Docs
`python train.py --train` - Trains the model
`python train.py --test` - Test the model on the dataset
`python train.py --gradio` - Run a gradio instance

### Options
`--shape` - The shape of the image. Defaults to 28
`--bias` - Defaults to 0
`--weights` - Defaults to weights.csv. Change the `defaultWeights` variable in train.py to `None` to start from scratch
`--circles` - (for `--train` and `--test` only) Directory for circles to train/test on. Defaults to `data/circles`
`--squares` - (for `--train` and `--test` only) Directory for squares to train/test on. Defaults to `data/squares`
`--output` - (for`--test` only) File to save weights to. Will not save weights otherwise.
`--count` - (for `--test` only) How many times to test the model on the dataset. Defaults to 1
`--host` - (for `--gradio` only) Host to run the gradio instance on. Defaults to 127.0.0.1
`--port` - (for `--gradio` only) Port to run the gradio instance on. Defaults to 8080