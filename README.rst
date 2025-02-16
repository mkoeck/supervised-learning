=========
Supervised Learning
=========

Supervised Learning is an Odoo module that provides an extensible framework for integrating machine learning models within Odoo. It enables the creation, training, and deployment of supervised learning pipelines for tasks such as classification and regression.

This module is currently in **prototype** stage, and additional estimators and functionalities will be added in the future.

**Key Features:**

- **Machine Learning Pipelines**: Define and manage machine learning workflows directly within Odoo.
- **Extensibility**: Easily add new models, preprocessing steps, and feature extraction methods.
- **Integration with Odoo Models**: Train models using existing Odoo data and apply predictions within business workflows.
- **Support for Multiple Estimators**: Provides a flexible framework to incorporate various machine learning algorithms.

**Table of contents**

.. contents::
   :local:

Usage
=====

This module introduces a framework for machine learning in Odoo but does not provide ready-to-use models. Users must configure datasets, define estimators, and train models within the module.

For now, available estimators include standard classification and preprocessing techniques, but more will be added.

Installation
============

⚠ WARNING ⚠

Installing the module in demo mode can take a long time because the entire 20newsgroups dataset is loaded as demo data.

See the `scikit-learn documentation on the 20 Newsgroups dataset <https://scikit-learn.org/stable/datasets/real_world.html#the-20-newsgroups-text-dataset>`_ for more details.

To install this module:

1. Ensure dependencies are installed:

   .. code-block:: bash

      pip install -r requirements.txt

2. Install the module in Odoo:
   
   - Go to **Apps** and search for "Supervised Learning"
   - Click **Install**

Configuration
=============

1. Define a **Dataset**:
   
   - Select an Odoo model as the data source.
   - Choose dependent and independent variables.
   - Configure preprocessing steps and feature extraction if needed.

2. Add **Estimators**:
   
   - Choose an algorithm (e.g., Random Forest, TF-IDF Vectorizer).
   - Configure hyperparameters.

3. Train the Model:
   
   - Click **Train Model** to initiate training.
   - Review model performance metrics (Precision, Recall, F1-score).

4. Apply Predictions:
   
   - By clicking **Create Contextual Action** you can add a server action to your chosen model (little cog icon). This server action lets you make predictions on your chosen model.

Changelog
=========

Next
~~~~

- Addition of more machine learning estimators
- Improved dataset handling
- Automated training and retraining options

Bug Tracker
===========

Bugs are tracked on `GitHub Issues <https://github.com/mkoeck/supervised-learning/issues>`_.
If you encounter an issue, please report it with detailed reproduction steps.

Credits
=======

Authors
~~~~~~~

* Michael Köck
