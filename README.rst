=============================
Cookiecutter Pytorch Template
=============================

A `Cookiecutter <https://github.com/audreyr/cookiecutter>`_ template for PyTorch projects.

.. contents:: Table of Contents
   :depth: 2

Requirements
============
* Python >= 3.8
* PyTorch >= 1.1
* Tensorboard >= 1.4

Features
========
* Clear folder structure which is suitable for many deep learning projects.
* Runs are configured via ``.yml`` files allowing for easy experimentation.
* Checkpoint saving and resuming.
* Tensorboard logging

Usage
=====

.. code::

    $ pip install cookiecutter
    $ cookiecutter https://github.com/khornlund/cookiecutter-pytorch
    $ cd path/to/repo

A template project has now been created! You can run the MNIST example using:

.. code::

    $ conda env create --file environment.yml
    $ conda activate <your-project-name>
    $ <your-project-name> train

Example Projects
================
Here are some projects which use this template:

1. `Severstal Steel Defect Detection (Kaggle) <https://github.com/khornlund/severstal-steel-defect-detection>`_
2. `Aptos Blindness Detection (Kaggle) <https://github.com/khornlund/aptos2019-blindness-detection>`_
3. `Understanding Cloud Organization (Kaggle) <https://github.com/khornlund/understanding-cloud-organization>`_

Custom Defaults
===============
If you fork this repo, you can modify ``cookiecutter.json`` to provide personalised defaults eg.
name, email, username, etc.

Acknowledgements
================
This template was based on `PyTorch Template <https://github.com/victoresque/pytorch-template>`_.
