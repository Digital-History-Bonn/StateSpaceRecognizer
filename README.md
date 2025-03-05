# State Space Resognizer: A SSM base OCR model using mamba-2 blocks

This repository contains the core components for the State Space Recognizer, utilizing mamba-2 blocks.
Text-line images are processed as 1D-Sequence and the output sequence is generated autoregressive.

The code has been tested on Python 3.10.

````
git clone https://github.com/Digital-History-Bonn/StateSpaceRecognizer.git && cd StateSpaceRecognizer
pip install .[gpu]
````
Additional options are [dev] and [newspaper]. For development with historical newspapers:

````
git clone https://github.com/Digital-History-Bonn/StateSpaceRecognizer.git && cd StateSpaceRecognizer
pip install .[dev, newspaper]
````
