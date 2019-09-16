# motion-structure-used-in-perception
Python code and download links to the data of
[Bill et al., "Hierarchical structure is employed by humans during visual motion perception" (2019)](https://www.biorxiv.org/content/10.1101/758573v1).

This repository allows you to:
* Generate figures 2, 3 and 4 from the main paper,
* Collect your own data,
* Run the full analysis pipeline (if you are willing to dig into the code, a bit).

In case of questions, please contact [Johannes Bill](mailto:johannes_bill@hms.harvard.edu).


## Table of contents

* [Installation](#installation)
* [Usage](#usage)
  + [Plot figures](#plot-figures)
  + [Collect your own data](#collect-your-own-data)
  + [Data download](#data-download)
  + [Data analysis](#data-analysis)
* [Miscellaneous](#miscellaneous)
  + [List of directories](#list-of-directories)
  + [Fonts](#fonts)

## Installation

We assume a Ubuntu-based Linux installation. On Mac, you should be able to homebrew with sip and pyqt.
In the cloned repository, we suggest to use a virtual environment with Python 3.6+:

```
$ python3 -m pip install --user --upgrade pip   # Install pip (if not yet installed)
$ sudo apt-get install python3-venv             # May be needed for environment creation
$ python3 -m venv env                           # Create environment
$ source env/bin/activate                       # Activate env
$ python3 -m pip install --upgrade pip          # Make sure the local pip is up to date
$ pip3 install wheel                            # Install wheel first
$ pip3 install -r requirements.txt              # Install other required packages
$ deactivate                                    # Deactivate env
```

## Usage

__Always__ start your session by running `source run_at_start.sh` and end it with `source run_at_end.sh`. These will set up the virtual environment and python path. Here are some cookbooks.

### Plot figures

Re-plotting the figures from the main paper is quick and easy:

```
$ source run_at_start.sh
$ cd plot
$ python3 plot_fig_2.py   # Plot Figure 2
$ python3 plot_fig_3.py   # Plot Figure 3
$ python3 plot_fig_4.py   # Plot Figure 4
$ cd ..
$ source run_at_end.sh
```

All figures will be saved in `./plot/fig/` as png and pdf.

### Collect your own data

#### MOT experiment

This experiment requires Python as well as MATLAB with Psychtoolbox.
Please make sure to have at least 2GB of disk space available per participant.
Questions on the data collection for the MOT experiment can also be directed to Hrag Pailian (pailian@fas.harvard.edu).

1. Generate trials:
  * `$ source run_at_start.sh`
  * `$ cd rmot/generate_stim`
  * Adjust `nSubjects=...` in file  `generate_trials_via_script.sh` to your needs.
  * Generate trials via `$ ./generate_trials_via_script.sh` (This may take a while depending on processor power.)
  * Resulting trials are written to:
    * `data/rmot/myexp/trials` for the Python data (will be needed for simulations and analyzes)
    * `data/rmot/myexp/matlab_trials` for the data collection with MATLAB
2. Run the experiment: For each participant `n=1,..`
  * Copy the content of `data/rmot/myexp/matlab_trials/participant_n/` into `rmot/matlab_gui/Trials/`.
  * `$ cd ../matlab_gui`
  * Determine the participant's speed via repeated execution of `Part_1_Thresholding.m` (will prompt for speed on start).
  * Conduct the main experiment via `Part_2_Test.m` (will prompt for speed and `n`).
  * Copy the saved responses to `data/rmot/myexp/responses/` and rename the file to `Response_File_Test_Pn.mat`.
3. Convert the data back to Python format:
  *  `$ cd ../ana`
  * For each participant `n=1,..`, run  
   `$ python3 convert_mat_to_npy.py data/myexp/responses/Response_File_Test_Pn.mat`.
  * `$ cd ../..`
  * `$ source run_at_end.sh`

Continue with the data analysis (see below).


#### Prediction experiment

This experiment is fully Python-based.

```
$ source run_at_start.sh
$ cd pred/gui
$ python3 play.py presets/example_trials/GLO.py -f -T 10   # EITHER: try out 10 trials (ca. 2 min)
$ ./run_full_experiment.sh -u 12345                        # OR: run the full experiment (ca. 75 min)
$ cd ../..
$ source run_at_end.sh
```
Continue with the data analysis (below).

If you run the full experiment, your data will be stored in `/data/pred/myexp/`.
Please refer to [`/pred/gui/README.md`](./pred/gui/README.md) for further information -- especially to ensure a stable frame rate before running a full experiment.


### Data download

The data from the publication can be downloaded here:

* MOT experiment (~445kB): https://ndownloader.figshare.com/files/17670059
* Prediction experiment (~282MB): https://ndownloader.figshare.com/files/17670065

For below analyses, unzip the content of these archives into the directories `data/rmot/paper` and `data/pred/paper` respectively. Then, execute steps 1. and 3. (replacing `myexp` with `paper`) in the description of _Collect your own data_ >> _MOT experiment_.

### Data analysis

Use the following analysis chain to recreate the aggregate data files provided in `/data` from the raw data in `/data/rmot/paper` and `/data/pred/paper` -- or to analyze your own data (see above). The analysis may require some understanding of the Python code. So, please, do not expect a direct copy-and-paste workflow.

#### MOT experiment
```
$ source run_at_start.sh
$ cd rmot/ana
```
1. Set up a _data set labels_ (DSL) file to link human data to simulation data:
  * You can use `DSLs_rmot_template.py` as a template.
  * Adjust `exppath` and `subjects`. Make sure `simpath` exists.
  * For each participant, create an entry block and enter the participant's `["speed"]` (from above 'thresholding').
  * The `["sim"]` entries will be filled later.
2. Set up the `config_datarun.py` file for simulations:
  * You can use `config_datarun_template.py` as a template.
  * Adjust the `import` to import from your DSL file and ensure that `cfg["global"]["outdir"]` exists.
  * Adjust `cfg["observe"]["datadir"]` to point to the (Python) trials.
  * You may want to reduce `reps_per_trial` from 25 to 1 to speed up the simulation (optional).
3. Prepare the simulations in `create_config_for_participant.py`:
  * Adjust lines `8-11` to match your DSLs, config, and trial directory.
4. Run observer models with different motion structure priors on the experiment trials:
  * For each participant and stimulus condition:
    - Adjust lines `6` and `7` in `create_config_for_participant.py`.
    - Run `$ ./start_datarun_script.sh`.
    - Enter the `DSL` of the simulation in your DSL file's `["sim"]` entry of the respective participant and condition.
    - Warning: The simulations may take a while (we used the HMS cluster).
  * Collect all results via `$ python3 load_human_and_sim_to_pandas.py` (adjust line `7`).
  * Copy the created `pkl.zip` file to the repository's `/data/` directory.
5. Plot the figure:
  * `$ cd ../../plot`
  * Adjust `fname_data=` to point to your data in `plot_fig_2.py`.
  * `$ python3 plot_fig_2.py   # Plot Figure 2`

```
$ cd ..
$ source run_at_end.sh
```

#### Prediction experiment

```
$ source run_at_start.sh
$ cd pred/ana
```
1. Run Kalman filters with different motion priors on the experiment trials:
  * In file `config_datarun_MarApr2019.py`, direct `cfg["observe"]["datadir"]` to the experiment data.
  * For each participant and stimulus condition:
    * In `config_datarun_MarApr2019.py`, enter `GROUNDTRUTH=` and `datadsl=`.
    * Run `$ python3 run.py config_datarun_MarApr2019`
    * Keep track of the _data set labels_ (DSLs) linking experiment and simulation data, in a file similiar to `DSLs_predict_MarApr2019.py`.
2. Fit all observer models (for Fig. 3):
  * Update the _parameters_ section in `fit_noise_models_with_lapse_from_DSLfile.py`, especially:  
    `exppath`, `outFilename`, and `import` from your DSL file.
  * `$ python3 fit_noise_models_with_lapse_from_DSLfile.py`
  * Copy the `outFilename` file to the repository's `/data/` directory.
3. Bias-variance analysis (for Fig. 4):
  * Update the _parameters_ section in `estimate_bias_variance.py`, especially:  
    `path_exp`, `outfname_data`, and `import` from your DSL file.
  * `$ python3 estimate_bias_variance.py`
  * Copy the `outfname_data` file to the repository's `/data/` directory.
4. Plot the figures:
  * `$ cd ../../plot`
  * Adjust `fname_data=` to point to your data in `plot_fig_3.py` and `plot_fig_4.py`.
  * `$ python3 plot_fig_3.py   # Plot Figure 3`
  * `$ python3 plot_fig_4.py   # Plot Figure 4`

```
$ cd ..
$ source run_at_end.sh
```

**Remark:** The provided code implements some small improvements in the numerical fitting (in step 2.) as compared to the _bioRxiv preprint v1_: because the log-likelihood landscape is non-convex, the optimizer can get stuck in local optima. The improved code uses a different method in `scipy.optimize.minimize` including gradients and yields even stronger significance in Fig. 3 as well as better model distinction in Fig. S2 than the _bioRxiv preprint v1_. Improved figures will be included in future versions of the preprint.

## Miscellaneous

### List of directories

* `data`: Experiment data and simulation/analysis results
* `pckg`: Python imports of shared classes and functions
* `plot`: Plotting scripts for Figures 2, 3 and 4
* `pred`: Simulation and analyis scripts for the prediction task
* `rmot`: Simulation and analyis scripts for the rotational MOT task

### Fonts

If the 'Arial' font is not installed already:

```
$ sudo apt-get install ttf-mscorefonts-installer
$ sudo fc-cache
$ python3 -c "import matplotlib.font_manager; matplotlib.font_manager._rebuild()"
```
...and if you really want it all: the stars in Figure 3 indicating significance use font type "FreeSans".

