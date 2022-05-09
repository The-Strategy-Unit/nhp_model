# New Hospitals Demand Model

## Environment setup

To run the model locally you will need to have installed miniconda, git, and VSCode. This is assuming that you are running on Windows, steps will need to be adjusted for other platforms.

1. Install [miniconda](https://conda.io/projects/conda/en/latest/user-guide/install/windows.html). Make sure to add miniconda to the path.
2. Clone the git repository:
  ```
  git clone https://github.com/The-Strategy-Unit/834_nhp_model.git
  ```
3. Open the repository in Visual Studio Code, then open a terminal `Terminal -> New Terminal (Ctrl+Shift+')`.
4. In the terminal, type `conda init`
5. In the terminal, type `conda env create -f environment.yml`.
6. Press `Ctrl+Shift+p`, then type `Select Interpreter`. Press the down key and select `Python 3.10.4 ('nhp')`.
7. Get a copy of the "data" directory and copy it into the folder.

## Running the model

The simplest way to test the model is to press `F5` in VS code. There are a number of debug profiles for testing each individual type of model (A&E, Inpatients, Outpatients), or to run all of the principal models, or run all of the models in parallel.

You can use all of the VS code debugging tools like breakpoints and the debug console. Consult the [VS code documentation](https://code.visualstudio.com/docs/python/debugging) for more on how to do this.

There is a Jupyter notebook, [`run_model.ipynb`](run_model.ipynb) which runs the models for a given params file.
