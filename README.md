# in-ear-stress
In-Ear Stress Classifier

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/danibene/in-ear-stress/add/pipe_pkl?labpath=example_predictions_onnx.ipynb)

This repository contains the code for a stress classifier using in-ear heartbeat sounds. The classifier is described in the paper "Stress classification with in-ear heartbeat sounds" by Danielle Benesch, Bérangère Villatte, Alain Vinet, Sylvie Hébert, Jérémie Voix, and Rachel E. Bouserhal. Physiological data used for training and evaluating the model may be made available upon reasonable request. Requests can be sent via email to the corresponding author at: `rachel.bouserhal@etsmtl.ca`

## Installation

Create a virtual environment and install the dependencies:

    $ python3 -m venv .venv
    $ source venv/bin/activate
    (.venv) $ pip install -r requirements.txt

## Usage

Run the example using the ONNX model file with example ECG data: `example_predictions_onnx.ipynb`.

