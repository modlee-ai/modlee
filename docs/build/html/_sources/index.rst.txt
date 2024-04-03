.. modlee documentation master file, created by
   sphinx-quickstart on Tue Feb 20 16:11:25 2024.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

modlee
======

.. toctree::
   :maxdepth: 4
   :caption: Getting started
   :hidden:

   
   README 
   README contents
   README/Installation
..   :ref:`README`
   .. contents::

.. toctree::
   :maxdepth: 10
   :caption: API Reference
   :hidden:

   modules/modlee

.. toctree::
   :maxdepth: 2
   :caption: Examples
   :hidden:

   notebooks/recommend
   notebooks/document

.. toctree::
   :maxdepth: 2
   :caption: Links 
   :hidden:

   modlee.ai <https://www.modlee.ai>
   modlee Dashboard <https://www.dashboard.modlee.ai>
   Discord <https://discord.com/channels/1205271955306192936/1205271956098646087>

Modlee is a machine learning tool that **documents** experiments for
reproduciblity and **recommends** neural network models suited for a
particular dataset. Modlee bypasses costly machine learning
experimentation by recommending performant models based on prior
experiments. Built on top of MLFlow, Modlee documents traditional
experiment assets (model checkpoints, (hyper)parameters, performance
metrics) and meta-features for
`meta-learning <https://ieeexplore.ieee.org/abstract/document/9428530>`__.
Based on these meta-features from prior experiments, Modlee recommends a
neural network model matched to a new task.

`Getting started <README.html>`_
======
Learn about modlee and how to get started.

`API Reference <py-modindex.html>`_
=========
Reference the source code and how to use the API.

Examples
========
View starter examples for :ref:`documentation<notebooks/document:Documentation>` or :ref:`recommendation<notebooks/recommend:Recommendation>`.

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
