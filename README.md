# [UNDER CONSTRUCTION]

# motion-structure-used-in-perception
Python code and experiment data for Bill et al. "Hierarchical structure is employed by humans during visual motion perception" (2019).


```
REMARK: We assume a Ubuntu-based linux installation.
        On Mac, you should be able to homebrew with sip and pyqt.
```

## Installation

We suggest to use a virtual environment with Python 3.6+:

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

If the 'Arial' font is not installed already:

```
$ sudo apt-get install ttf-mscorefonts-installer
$ sudo fc-cache
$ python3 -c "import matplotlib.font_manager; matplotlib.font_manager._rebuild()"
```

## List of directories

* `data`: Experiment data and simulation results
* `pckg`: Python package with shared classes and functions
* `plot`: Plotting scripts for Figures 2, 3 and 4
* `pred`: Simulation and analyis scripts for the prediction task
* `rmot`: Simulation and analyis scripts for the rotational MOT task

## Usage

__Always__ start your session by running `source run_at_start.sh` and end it with `source run_at_end.sh`. These will set up the virtual environment and python path. Here are some cookbooks.

### Plot figures

```
$ cd plot
$ python3 plot_fig_2.py   # Plot Figure 2
$ python3 plot_fig_3.py   # Plot Figure 3
$ python3 plot_fig_4.py   # Plot Figure 4
```

All figures will be saved in './plot/fig/' as png and pdf.

### Collect your own data

### Analyze your own data

