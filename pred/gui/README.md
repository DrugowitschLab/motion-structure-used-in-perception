# Structured Motion Stimuli

A python script to explore structured motion stimuli for location prediction experiments.

## Usage

Play structured motion stimuli via:

```
$ python3 play.py config_file_for_stimulus.py [optional arguments]
```

Optional arguments:
```
 -f                    # Full screen mode (exit via ESC)
 -T num trials         # Maximum number of trials (default: infinity)
 -R num reps           # Trial repetitions (requires -T; leads to T/R unique trials; default: 1)
 -g str                # Greeter displayed before first trial
 -u int                # Integer-valued ID of the participant
```

### Examples

1. Display a stimulus until the windows is closed (press ESC):
```
$ python3 play.py presets/example_trials/GLO.py -f
```

2. Run 5 unique trials, each repeated twice, (i.e., 10 trials in total) in full screen:
```
$ python3 play.py presets/example_trials/GLO.py -f -T 10 -R 2
```

3. Run a full experiment with 100 trials from each motion condition (pick a max. 5 digit ID; here: 12345):
```
$ ./run_full_experiment.sh -u 12345
```

## Advanced Installation

### Unstable frame rates (selecting the right backend)

The script has been tested with the TkAgg, Qt4Agg and Qt5Agg backends. I observed the default matplotlib backend `TkAgg` to work fine on some systems. On other systems, however, the frame rate is not stable. You can test for a stable frame rate by running **Example 1**. After completion, an evaluation of frame timings is printed. Frames are expected to be plotted with better-than-10ms precision. If the average interval significantly deviates from the target or has a high standard deviation, please select a Qt-based backend in `general_config.py`:

```
  backend_interactive = "Qt4Agg" or "Qt5Agg"
```

* *Linux*: The debian package with Qt4 Python bindings is `python3-pyqt4`.
```
$  sudo apt-get install python3-pyqt4
```

* *Mac*: Use homebrew to install sip and pyqt:
```
$ brew install sip --with-python3
$ brew install pyqt --with-python3
```
* *Windows*: Sorry, you're on your own.

## Authors

* **Johannes Bill** - *Initial work* - [Harvard Medical School](https://drugowitschlab.hms.harvard.edu/people-0) - johannes_bill@hms.harvard.edu


## Acknowledgments

* Thanks to Jan Drugowitsch and Sam Gershman for valuable feedback and discussions for the stimulus design.
* Thanks to Luke Rast for testing the program and backends on Mac.


## Version history

1.4 (2019-01-24) Repetitions of trials (via rng seedlist)  
1.3 (2019-01-03) Greeter, UserID, performance points, optimize mouse pointer  
1.2 (2018-10-30) Experiment automation (phases) and data storage  
1.1 (2018-10-04) Added colors, full screen, hide targets, play/pause & removed alpha_func  
1.0 (2018-10-04) Basic version (derived from chick demostim 1.5)  
