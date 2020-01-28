# TensorNetworks
Theory and code to analyze tensor networks for machine learning applications. 

This project is developed by the [Physics and Machine Learning Group](https://sites.google.com/s/185Lxg3icUi3qxMaMur9WNckY_kCIHfgs/p/1VgB8YLjVhMoYLXh7L1kDSS4GV7Fe6kdj/edit) of the [National University of Colombia](http://unal.edu.co/)

These are calculations to validate the article named: [A Generalized Language Model in Tensor Space](https://www.semanticscholar.org/paper/A-Generalized-Language-Model-in-Tensor-Space-Zhang-Zhang/fa744ef316f58139506f36bb3504ba5b27301918).
The model is developed in python using the tensor network library. There are the following tasks:
1. The import/ directory contains the raw data and initial preparation.
2. The src/ directory contains the code for the language representation of the corpus and the tensor representation for sentences. Adittionaly, we validate the results with respect Recurrent Neural Networks.
3. The output/ directory contains the results of running the model and plot of validation tests.
4. The write/ directory combines the estimates and graphs in an RMarkdown document.
