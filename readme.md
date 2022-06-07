# New Hospitals Demand Model

## Environment setup

To run the model locally you will need to have installed miniconda, git, and VSCode. This is assuming that you are running on Windows, steps will need to be adjusted for other platforms.

### Install

1. Install [miniconda](https://conda.io/projects/conda/en/latest/user-guide/install/windows.html). When installing, choose to install for all users, and check the box on the final page to add conda to the System PATH.
2. Install [VS Code](https://code.visualstudio.com/). You can accept the default values while installing.
3. Install [Git](https://git-scm.com/downloads). When asked to choose the default editor, you should pick a "VS Code" or "notepad", unless you know "vim".

### Set up VS Code

4. Open Visual Studio Code, on the left hand panel, press the "extensions" button (or press Ctrl+Shift+X)
5. Search for "python" and press install
6. Search for "jupyter" and press install
7. Clone the git repository: the easiest way is to press `Ctrl+Shift+p` and type "clone", you should see "Git: clone". Choose that, then choose "Clone from GitHub". Follow the prompts to sign in, then when it asks you to type a repository name, type `The-Strategy-Unit/834_nhp_model`.

### Set up Python

8. Open the repository in Visual Studio Code, then open a terminal `Terminal -> New Terminal (Ctrl+Shift+')`.
9. In the terminal
  ``` py
  conda init
  conda env create -f environment.yml
  ```
9. Press `Ctrl+Shift+p`, then type `Select Interpreter`. Press the down key and select `Python 3.10.4 ('nhp')`.
10. Get a copy of the `data.zip` archive, and extract the contents to the project's folder.

## Running the model

The simplest way to test the model is to press `F5` in VS code. There are a number of debug profiles for testing each individual type of model (A&E, Inpatients, Outpatients), or to run all of the principal models, or run all of the models in parallel.

You can use all of the VS code debugging tools like breakpoints and the debug console. Consult the [VS code documentation](https://code.visualstudio.com/docs/python/debugging) for more on how to do this.

There is a Jupyter notebook, [`run_model.ipynb`](run_model.ipynb) which runs the models for a given params file.
