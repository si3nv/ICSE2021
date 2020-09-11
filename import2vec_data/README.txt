This folder contains supplemental material to the "import2vec" submission to MSR 2019.

The folder contains a Jupyter notebook to run the examples described in the evaluation section:

    - Evaluation.ipynb

You can modify the examples to experiment with other libraries. In the case of Java, a helper function 'find_library()' should be used to find the exact name of a library we support in the model.

Along with the notebook, we also provide trained vectors for multiple dimensions for the 3 ecosystems:

Java: java_w2v_dim*.txt.gz      - dimensions 40, 50, 60 and 100
JavaScript: js_w2v_dim*.txt.gz  - dimensions 40, 50, 60, 100 and 200
Python: python_w2v_dim*.txt.gz  - dimensions 40, 50, 60, 100 and 200


To run the examples:

1/ assuming an anaconda install of python 3.6 or higher (https://www.anaconda.com/download/)

2/ create an environment:

    conda create --name import2vec

3/ activate the environment:

    source activate import2vec

4/ install some dependent libraries

    conda install jupyter
    conda install gensim

5/ start Jupyter in the directory of the Evaluation.ipynb notebook

    jupyter notebook

6/ if you run this on your local system, it should open a browser window for your Jupyter setup. If you don't go to the URL printed at the command line

7/ select "Evaluation.ipynb"

You're all set. Have fun exploring the models!