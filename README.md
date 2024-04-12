# nipystats
A python package for fMRI statistical analysis.

# TL;DR
This is basically a nilearn wrapper for first- and second-level analyzes. You need fMRIPrep outputs and configuration files to specify the models and the contrasts.
Probably similar package: [fitlins](https://github.com/poldracklab/fitlins).

# Installation

```
git clone https://github.com/ln2t/nipystats.git
cd nipystats
git checkout 2.1.0
virtualenv venv
source venv/bin/activate
python -m pip install .
```

Test install:

```
nipystats -h
```

# Configuration files examples

## First-level

You need a configuration file,  `participant_level-config.json`, typically
```
{
 "model": "modelname1",
 "model_description": "My model description",
 "trials_type": [
  "trial1",
  "trial2"
 ],
 "contrasts": [
   "trial1",
   "trial2-trial1"
 ],
 "tasks": [
  "taskname"
 ],
 "first_level_options": {
  "slice_time_ref": 0.5,
  "smoothing_fwhm": 0,
  "noise_model": "ar1",
  "signal_scaling": [
   0,
   1
  ],
  "hrf_model": "SPM",
  "high_pass": 0.0078125,
  "drift_model": "polynomial",
  "minimize_memory": false,
  "standardize": false
 },
 "concatenation_pairs": null
}
```
Command-line call:
```
nipystats /path/to/bids/rawdata /path/to/derivatives participant --fmriprep_dir /path/to/fMRIPrep/dir --config /path/to/participant_level_config.json
```

## Second-level 
You need a configuration file,  `group_level-config.json`, typically
```
{
 "model": "My model name for group analysis",
 "covariates": ['age', 'group'],
 "contrasts": [
  "group1-group2"
 ],
 "tasks": [
  "taskname"
 ],
 "first_level_model": "modelname1",
 "first_level_contrast": "trial2-trial1",
 "concat_tasks": null,
 "add_constant": false,
 "smoothing_fwhm": 8,
 "report_options": {
   "plot_type": "glass",
   "height_control": "fdr"
 },
 "paired": false,
 "task_weights": null
}
```
Command-line call:
```
nipystats /path/to/bids/rawdata /path/to/derivatives group --fmriprep_dir /path/to/fMRIPrep/dir --config /path/to/group_level-config.json
```

# Disclaimer

I am not a professional coder, so don't get mad if you can't stomach my "style" :-)
Also, this software comes "as is", and I do not guarantee that it will work.
You are most than welcome to open a pull request if you feel, or simply post your questions/issues here.

# More explainations, at last!

## Why and what

This package is developped as an alternative to stuff like [fitlins](https://github.com/poldracklab/fitlins), which honestly I love BUT I had a lot of trouble to understand how to make configuration files for my own analyzes.
I therefore ended up writing my own stuff and figured I could also share it!

The basic principles are:
- it is a BIDS app, which means in particular that:
 A. it (should) work BIDS datasets,
 B. the command-line arguments are somehow standardized: `nipystats rawdata derivatives participants [options]` or `nipystats rawdata derivatives group [options]`,
- it works only for data preprocessed with [fMRIPrep](https://github.com/nipreps/fmriprep),
- it heavily relies on the beautiful [nilearn](https://nilearn.github.io/stable/index.html) package,
- the derivatives ("outputs") contain `html` report that you can easily read and share with your friends (potentially: colleagues).

## Required dataset

Let's assume you have a BIDS dataset, contained in a folder that for the sake of simplicity we will refer to the `rawdata`. So `rawdata` typically has the following stuff in it:
```
rawdata/dataset_description.json
rawdata/participants.tsv
rawdata/sub-01
rawdata/sub-02
rawdata/sub-03
...
```
The meaning of these things are explained in the [BIDS documentation](https://bids-specification.readthedocs.io/en/stable/). Of course, each subject should have functionnal scans (if not you're probably not on the page you were looking for...).

We also need fMRIPrep outputs, which we suppose are located in `derivatives/fmriprep`. Note that some people tends to have the `derivatives` next to the `sub-01, sub-02, sub-03` folders - I personnaly find it very inconvenient (think when you want to read an `html` report in the derivatives but have a lot of participants: scroll scroll scroll...). But in the end it doesn't matter much. So we have something like:

```
derivatives/fmriprep/dataset_description.json
derivatives/fmriprep/sub-01
derivatives/fmriprep/sub-01.html
derivatives/fmriprep/sub-02
derivatives/fmriprep/sub-02.html
derivatives/fmriprep/sub-03
derivatives/fmriprep/sub-03.html
...
```

Note also that for better control on what you're doing, it is sometimes convenient to tag you fmriprep folder name with the version used, for instance mine looks like `derivatives/fmriprep_23.1.3`. This information is also located in `derivatives/fmriprep/dataset_description.json` but having this explicitly in the folder name avoid some merging/overwriting between versions... that's up to you in the end!

## How typical analyses work in fMRI

(work in progress)

### Participant level (First-level)

### Group level (Second-level)

## The configuration files

Those are `json` files (so basically a text file with a special syntax that makes it easy to read and write both for humans and robots).



