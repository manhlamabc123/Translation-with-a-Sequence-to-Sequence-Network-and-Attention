#  Chatbot using Neural Network

<details>
  <summary>Table of Contents</summary>
  <ol>
    <li><a href="#purpose-of-project">Purpose of Project</a></li>
    <li><a href="#architecture">Architecture</a></li>
    <li><a href="#technologies">Technologies</a></li>
    <li><a href="#how-to-run">How to run</a></li>
    <li><a href="#acknowledgments">Acknowledgments</a></li>
  </ol>
</details>

## Purpose of Project

* Train a model to translate from French to English
* Learn about Sequence-to-Sequence Model
* Learn about Attention

## Architecture

* Encoder:

![Alt text](images/encoder-network.png "Title")
* Decoder:

## Technologies

* Language: Python
* Framework: PyTorch

## How to run

### Prerequisites

Required Libraries:
* pytorch
* numpy
* nlkt

### Data preparation

Make sure everything is set in `intents.py` file.

### To train the model
```
python train.py
```

### To chat
```
python chat.py
```

## Acknowledgments

* Youtube Tutorial: [video](https://www.youtube.com/watch?v=yN7qfBhfGqs)
* How to Implement Spatial Transformer Network in PyTorch: [doc](https://pytorch.org/tutorials/intermediate/spatial_transformer_tutorial.html)
