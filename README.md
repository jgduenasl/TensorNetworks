# Tensor Networks
The goal of this work is apply the tensor network formalism to machine learning. This repository is composed of different projects starting in the implementation of the formalism of the DMRG algorithm, and going on with applications to physical systems described by Matrix Product State (MPS). Besides this, we explore applications in Machine Learning (ML) and Natural Language Processing (NLP). This project is developed by the [Physics and Machine Learning Group](https://sites.google.com/s/185Lxg3icUi3qxMaMur9WNckY_kCIHfgs/p/1VgB8YLjVhMoYLXh7L1kDSS4GV7Fe6kdj/edit) of the [National University of Colombia](http://unal.edu.co/)

Each project follows the methodolgy developed by the Human Rights Data Analysis Group [HRDAG](https://hrdag.org/2020/01/07/learning-a-modular-auditable-and-reproducible-workflow/). There is a directory where each project is developed. The scripts are typed in make and python using the numpy, pandas, tensor network or pytorch libraries. 

The directories structure into each project is given by the required tasks. A common structure is:

1. The import/ directory contains the raw data, the cleaning and wrangling preparation of data and a suitable files to perform analysis.
2. The task1/ directory contains the cleaning data, the implementation of the specific task to be performed.
3. The output/ directory contains the results of running the tasks and the output files produced in the analysis.
4. The write/ directory combines the results and graphs in a Markdown document.
