import numpy as np
from PIL import Image
import csv
import os
from itertools import cycle
from shutil import get_terminal_size
from threading import Thread
from time import sleep
import argparse
import gradio as gr


class Loader:
    def __init__(self, desc="Loading...", end="Done!", timeout=0.1):
        """
        A loader-like context manager

        Args:
            desc (str, optional): The loader's description. Defaults to "Loading...".
            end (str, optional): Final print. Defaults to "Done!".
            timeout (float, optional): Sleep time between prints. Defaults to 0.1.
        """  # noqa: E501
        self.desc = desc
        self.end = end
        self.timeout = timeout

        self._thread = Thread(target=self._animate, daemon=True)
        self.steps = ["⢿", "⣻", "⣽", "⣾", "⣷", "⣯", "⣟", "⡿"]
        self.done = False

    def start(self):
        self._thread.start()
        return self

    def _animate(self):
        for c in cycle(self.steps):
            if self.done:
                break
            print(f"\r{self.desc} {c}", flush=True, end="")
            sleep(self.timeout)

    def __enter__(self):
        self.start()

    def stop(self):
        self.done = True
        cols = get_terminal_size((80, 20)).columns
        print("\r" + " " * cols, end="", flush=True)
        print(f"\r{self.end}", flush=True)

    def __exit__(self, exc_type, exc_value, tb):
        # handle exceptions with those variables ^
        self.stop()


defaultCircles = "data/circles"
defaultSquares = "data/squares"
defaultTestCircles = "data/circles"
defaultTestSquares = "data/squares"
defaultShape = 28
defaultBias = 0
defaultWeights = "weights.csv"

done = False


def train(
    circles=defaultCircles,
    squares=defaultSquares,
    shape=defaultShape,
    bias=defaultBias,
    weights=defaultWeights,
    output=defaultWeights,
    count=1,
):
    cfiles = len(os.listdir(circles))
    sfiles = len(os.listdir(squares))
    files = int(min(cfiles, sfiles))
    weights = weights

    weightsloader = Loader("Loading weights...", "Done!", 0.05).start()

    if type(weights) == str:
        with open(weights, "r") as f:
            print("\nExisting weights found. Loading...")
            weights = []
            reader = csv.reader(f)
            for row in reader:
                # Check if row isn't empty
                if row != []:
                    for i in range(shape):
                        for j in range(shape):
                            weights.append(float(row[i * shape + j]))
    else:
        print("\nNo weights found. Setting all weights to 0...")
        weights = np.zeros(int(shape * shape))

    weightsloader.stop()

    def predictFromActivations(activations, weights, bias):
        output = np.dot(activations, weights)
        if output < bias:
            return ["circle", output]
        elif output >= bias:
            return ["square", output]

    def trainFromActivations(activations, desiredShape, weights, bias):
        shape, output = predictFromActivations(activations, weights, bias)
        returnval = None

        if shape == desiredShape:
            returnval = 1
        else:
            # Update weights
            for i in range(len(weights)):
                if output < bias:
                    weights[i] = weights[i] + activations[i]
                else:
                    weights[i] = weights[i] - activations[i]
            returnval = 0

        return returnval

    trainloader = Loader("Training...", "Successfully trained!", 0.05).start()

    for i in range(count):
        for file in range(int(files)):
            square = np.array(
                Image.open(squares + "/" + os.listdir(squares)[file])
            )  # noqa: E501
            circle = np.array(
                Image.open(circles + "/" + os.listdir(circles)[file])
            )  # noqa: E501

            if len(circle.shape) == 3:
                circle = np.dot(circle, [0.299, 0.587, 0.114])
            if len(square.shape) == 3:
                square = np.dot(square, [0.299, 0.587, 0.114])

            # Scale each activation to be between 0 and 1
            circle = circle / 255
            square = square / 255

            # Make activations 1d array
            circle = circle.flatten()
            square = square.flatten()

            # Train
            trainFromActivations(circle, "circle", weights, bias)
            trainFromActivations(square, "square", weights, bias)

    trainloader.stop()

    if output:
        outputloader = Loader("Saving weights...", "Done!", 0.05).start()
        with open(output, "w") as f:
            writer = csv.writer(f)
            writer.writerow(weights)
        outputloader.stop()


def test(
    circles=defaultTestCircles,
    squares=defaultTestSquares,
    shape=defaultShape,
    bias=defaultBias,
    weights=defaultWeights,
):
    cfiles = len(os.listdir(circles))
    sfiles = len(os.listdir(squares))
    files = int(min(cfiles, sfiles))
    weights = weights

    weightsloader = Loader("Loading weights...", "Done!", 0.05).start()

    if type(weights) == str:
        with open(weights, "r") as f:
            print("\nExisting weights found. Loading...")
            weights = []
            reader = csv.reader(f)
            for row in reader:
                # Check if row isn't empty
                if row != []:
                    for i in range(shape):
                        for j in range(shape):
                            weights.append(float(row[i * shape + j]))
    else:
        print("\nNo weights found. Setting all weights to 0...")
        weights = np.zeros(int(shape * shape))

    weightsloader.stop()

    def predictFromActivations(activations, weights, bias):
        output = np.dot(activations, weights)
        if output < bias:
            return ["circle", output]
        elif output >= bias:
            return ["square", output]

    accuracy = 0
    totalfiles = 0
    for file in range(files):
        totalfiles += 2
        square = np.array(
            Image.open(squares + "/" + os.listdir(squares)[file])
        )  # noqa: E501
        circle = np.array(
            Image.open(circles + "/" + os.listdir(circles)[file])
        )  # noqa: E501

        if len(circle.shape) == 3:
            circle = np.dot(circle, [0.299, 0.587, 0.114])
        if len(square.shape) == 3:
            square = np.dot(square, [0.299, 0.587, 0.114])

        # Scale each activation to be between 0 and 1
        circle = circle / 255
        square = square / 255

        # Make activations 1d array
        circle = circle.flatten()
        square = square.flatten()
        # Predict
        prediction = predictFromActivations(circle, weights, bias)

        if prediction[0] == "circle":
            accuracy += 1

        prediction = predictFromActivations(square, weights, bias)

        if prediction[0] == "square":
            accuracy += 1

    # Convert accuracy to percentage
    percentage = accuracy / totalfiles
    print(
        str(percentage * 100) + "% accuracy on " + str(totalfiles) + " images."
    )  # noqa: E501


def gradioDemo(
    shape=defaultShape,
    bias=defaultBias,
    weights=defaultWeights,
    host="127.0.0.1",
    port=8080,
):
    weights = weights

    weightsloader = Loader("Loading weights...", "Done!", 0.05).start()

    if type(weights) == str:
        with open(weights, "r") as f:
            print("\nExisting weights found. Loading...")
            weights = []
            reader = csv.reader(f)
            for row in reader:
                # Check if row isn't empty
                if row != []:
                    for i in range(shape):
                        for j in range(shape):
                            weights.append(float(row[i * shape + j]))
    else:
        print("\nNo weights found. Setting all weights to 0...")
        weights = np.zeros(int(shape * shape))

    weightsloader.stop()

    def predictFromActivations(activations, weights, bias):
        output = np.dot(activations, weights)
        if output < bias:
            return ["circle", output]
        elif output >= bias:
            return ["square", output]

    # Create gradio instance
    def sketch_recognition(img, weights=weights, bias=bias):
        # create activation
        image = (np.array(img) / 255).flatten()

        # predict
        prediction = predictFromActivations(image, weights, bias)

        return prediction[0]

    sp = gr.Sketchpad(shape=(shape, shape), brush_radius=0.1)
    gr.Interface(fn=sketch_recognition, inputs=sp, outputs="text").launch(
        server_name=host, server_port=port
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--train",
        action="store_true",
    )
    parser.add_argument(
        "--test",
        action="store_true",
    )
    parser.add_argument(
        "--gradio",
        action="store_true",
    )
    parser.add_argument(
        "--shape",
        type=int,
    )
    parser.add_argument(
        "--bias",
        type=int,
    )
    parser.add_argument(
        "--weights",
        type=str,
    )
    parser.add_argument(
        "--circles",
        type=str,
    )
    parser.add_argument(
        "--squares",
        type=str,
    )
    parser.add_argument(
        "--output",
        type=str,
    )
    parser.add_argument(
        "--count",
        type=int,
    )
    parser.add_argument(
        "--host",
        type=str,
    )
    parser.add_argument(
        "--port",
        type=int,
    )
    args = parser.parse_args()

    if args.train:
        train(
            shape=args.shape if args.shape else defaultShape,
            bias=args.bias if args.bias else defaultBias,
            weights=args.weights if args.weights else defaultWeights,
            circles=args.circles if args.circles else defaultCircles,
            squares=args.squares if args.squares else defaultSquares,
            output=args.output if args.output else False,
            count=args.count if args.count else 1,
        )
    elif args.test:
        test(
            shape=args.shape if args.shape else defaultShape,
            bias=args.bias if args.bias else defaultBias,
            weights=args.weights if args.weights else defaultWeights,
            circles=args.circles if args.circles else defaultCircles,
            squares=args.squares if args.squares else defaultSquares,
        )
    elif args.gradio:
        gradioDemo(
            shape=args.shape if args.shape else defaultShape,
            bias=args.bias if args.bias else defaultBias,
            weights=args.weights if args.weights else defaultWeights,
            host=args.host if args.host else "127.0.0.1",
            port=args.port if args.port else 8080,
        )
