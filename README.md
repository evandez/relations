# Relations in Transformer LMs

This repository houses ongoing experiments that study how transformer LM implement relations.

## Setup

All code is tested on `MacOS Ventura (>= 13.1)` and `Ubuntu 20.04` using `Python >= 3.10`. It uses a lot of newer Python features, so the Python version is a strict requirement.

To run the code, create a virtual environment with the tool of your choice, e.g. conda:
```bash
conda create --name relations python=3.10
```
Then, after entering the environment, install the project dependencies:
```bash
python -m pip install invoke
invoke install
```
