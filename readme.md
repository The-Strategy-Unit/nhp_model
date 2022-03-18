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

The simplest way to test the model is to press `F5` in VS code, this will perform a single run of the model. You can use all of the VS code debugging tools like breakpoints and the debug console. Consult the [VS code documentation](https://code.visualstudio.com/docs/python/debugging) for more on how to do this.

You can also run the model directly from the command line by running

```
python run_model.py data/synthetic/test/20220110_104353 0 1 --debug
```

Here you pass the path to where the `params.json` file is stored, the 0 is to start running this model from the 0th (1st) model run, and the 1 indicates to run 1 model iterations. These two parameters are largely ignored if you pass the `--debug` flag, which causes the model to just run a single iteration.

If you remove the --debug flag the model will run for as many iterations as you ask for, but will run in parallel. Instead of displaying the results it will save the results to parquet files in `data/synthetic/test/20220110_104353/results`.

There is a Jupyter notebook, [`run.ipynb`](run.ipynb) which shows some more details on how to run the model and use the results, and an R script, [`demo.R`](demo.R) which shows how you can use the results in R.