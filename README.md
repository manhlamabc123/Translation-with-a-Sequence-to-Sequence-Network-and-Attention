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

![Alt text](images/attention-decoder-network.png "Title")

## Technologies

* Language: Python
* Framework: PyTorch

## How to run

### Prerequisites

Required Libraries:
* pytorch
* numpy

### To train and test the model
```
python main.py
```

## Acknowledgments

* Tutorial: [webpage](https://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html)