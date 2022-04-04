# New Hospitals Demand Model

## Environment setup

To run the model locally you will need to have installed miniconda, git, and VSCode. This is assuming that you are running on Windows, steps will need to be adjusted for other platforms.

1. Install [miniconda](https://conda.io/projects/conda/en/latest/user-guide/install/windows.html).
2. Clone the git repository:
  ```
  git clone https://github.com/The-Strategy-Unit/834_nhp_model.git
  ```
3. Open the repository in Visual Studio Code, then open a terminal `Terminal -> New Terminal (Ctrl+Shift+')`.
4. In the terminal, type `conda init`
5. In the terminal, type `conda env create -f environment.yml`.
6. Press `Ctrl+Shift+p`, then type `Select Interpreter`. Press the down key and select `Python 3.8.12 ('nhp')`.
7. Get a copy of the "data" directory and copy it into the folder.

## Running the model

The simplest way to test the model is to press `F5` in VS code. There are a number of debug profiles for testing each individual type of model (A&E, Inpatients, Outpatients), or to run all of the principal models, or run all of the models in parallel.

You can use all of the VS code debugging tools like breakpoints and the debug console. Consult the [VS code documentation](https://code.visualstudio.com/docs/python/debugging) for more on how to do this.

There is a Jupyter notebook, [`run_model.ipynb`](run_model.ipynb) which runs the models for a given params file. It will create a folder to store the results in the form `data/[DATASET]/results/[SCENARIO]/[RUNTIME]`.

You can also run the model directly from the command line by running

```
python run_model.py data/[DATASET]/results/[SCENARIO]/[RUNTIME] 0 1 --debug --type=[ACTIVITY_TYPE]
```

Here you pass the path to where the `params.json` file is stored, the 0 is to start running this model from the 0th (1st) model run, and the 1 indicates to run 1 model iterations. The second of these two parameters is ignored if you pass the `--debug` flag, which causes the model to just run a single iteration.

`ACTIVITY_TYPE` should be one of `"aae", "ip", "op"`.

The rest of the arguments in the file path need to be setup before hand, e.g. with [`run_model.ipynb`](run_model.ipynb).

If you remove the --debug flag the model will run for as many iterations as you ask for, but will run in parallel. Instead of displaying the results it will save the results to parquet files in `data/[DATASET]/[SCENARIO]/[RUNTIME]/results`.
