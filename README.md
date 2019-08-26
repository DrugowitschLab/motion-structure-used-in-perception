# motion-structure-used-in-perception
Python code and experiment data for Bill et al. "Hierarchical structure is employed by humans during visual motion perception" (2019).


```
REMARK: We assume a ubuntu-based linux installation.
        On Mac, you should be able to use homebrew with sip and pyqt.
```

## Installation

We will use a virtual environment with Python 3.6+:

```
$ python3 -m pip install --user --upgrade pip   # Install pip (if not yet installed)
$ python3 -m venv env                           # Create environment
$ source env/bin/activate                       # Activate env
$ pip3 install --user -r requirements.txt       # Install required packages
$ deactivate                                    # Deactivate env
```

## List of directories

* `data`: Experiment data and simulation results
* `pckg`: Python package with shared classes and functions
* `plot`: Plotting scripts for Figures 2, 3 and 4
* `pred`: Simulation and analyis scripts for the prediction task
* `rmot`: Simulation and analyis scripts for the rotational MOT task

## Usage

__Always__ start your session by running `./run_at_start.sh` and end it with `./run_at_end.sh`. These will set up the virtual environment and python path. Here are some cookbooks.

### Plot figures

### Collect your own data

### Analyze your own data

