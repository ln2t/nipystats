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
(etc)
```

Note also that for better control on what you're doing, it is sometimes convenient to tag you fmriprep folder name with the version used, for instance mine looks like `derivatives/fmriprep_23.1.3`. This information is also located in `derivatives/fmriprep/dataset_description.json` but having this explicitly in the folder name avoid some merging/overwriting between versions... that's up to you in the end!

## How typical analyses work in fMRI

For the sake of simplicity, we assume that there is only one task and one session per participant in the dataset.
As per BIDS, the events of the task must be described in the `events.tsv` file, which contains the onset and duration of the events (note that there is an optional column named `modulation` - see BIDS documentation for details -  but know that this is also supported in `nipystats`.)
A typical file would look like:
```
onset	duration	trial_type
30	30              right_finger
90	30              right_finger
150	30              right_finger
210	30              right_finger
270	30              right_finger
330	30              right_finger
```
This is a simple example where trials are of one type (`right_finger`). The units are in seconds.
Now, you might have more than one type of trial types; for instance, if you participant also hears some audio instructions, you might have something like:
```
onset	duration	trial_type
30	0               audio_instructions
30	30		right_finger
90	30              right_finger
150	30              right_finger
210	0               audio_instructions
210	30              right_finger
270	30              right_finger
330	30              right_finger
330	0               audio_instructions
```
(Here the paradigm is just for illustration purposes, don't attempt to find logic in this).
The first step in your statistical analysis will be to define the *model* for the data.
We do not include here an even light introduction to the subject, but let's recall that to build the model, you can typically consider *all* trials to define regressors (those are the columns in the design matrix).
These regressors are typically convolved with the HRF, and the design matrix is completed by added a set of confounds (which are basically other regressors that were not convolved with the HRF - these include motion parameters, typically).
Heavily relying on nice tools in `nilearn`, `nipystats` take the `events.tsv` file and will automatically build these regressors.
The "novelty" in `nipystats` is that is allows you to (conveniently) *select* which type of trials you want to include.
For instance, if for some reason you want one the `right_finger` events to converted in a column for your design matrix, then you can do so by specifying the following in the configuration file:

```
(...)
"trials_type": [
  "right_finger"
 ]
(...)
```

On the other hand, if you want to use also the `audio_instructions` trials, then you could use:
```
(...)
"trials_type": [
  "right_finger",
  "audio_instructions"
 ]
(...)
```
`nipystats` basically filters the `events.tsv` file by the entries in there.

Regarding the confounds, `nipystats` makes some reasonable choice for you, taking one column to model the scanner drift as well as the 6 motion parameters.
Some of the parameters in the config file allow you to somewhat tune this, see the `first_level_options` field. Those are essentially passed to the `nilearn` first-level GLM functions.

Once the data has been fitted, one must choose contrast vectors. These heavily depend on your research question, and in `nipystats` you can define them using string with operations like + and - (again, this uses `nilearn` machinery).
So typically, if we want to test for the `right_finger` effect, we choose:

```
(...)
"contrasts": [
  "right_finger"
 ]
(...)
```
 
If, for some fancy reasons, you want to have a contrast that mixes different types of trials, you can do it as follows:
```
(...)
"contrasts": [
  "right_finger+audio_instructions"
 ]
(...)
```
Again, this is just an example to understand how to build your own configuration files. Of course the field `contrast` is a list, so you can have several contrasts by separating them with a coma:
```
(...)
"contrasts": [
  "right_finger+audio_instructions",
  "right_finger",
  "audio_instructions-right_finger",
 ]
(...)
```

The full configuration is labeled by a "model name", and this is going to be used to build the outputs.
This way, you can run several models on your data, and `nipystats` will not mix the outputs together (provided of course you use different model names, duh).
Moreover, the actual outputs (statistical maps and reports) also contain the model name, so that if you share one file with a friend, the reference to the model will still be there.
The configuration file is also saved (copied) to the output folder, so if you lose yours, or just share the derivative folder, people will be able to re-run your analysis - a feature that is really relevant in the context of reproducibility.
Note that the reports, generated using the `nilearn` routines, is a single `html` for all contrasts (they appear in different sections), while the maps (e.g. statistics or effect size maps - the "betas") are saved for each contrast.
The full name of these maps thus also contain the contrast name, so that we know what we are sharing/looking at.
Finally, `nipystats` also produces cluster tables (for each contrast). These cluster tables are mostly made by `nilearn`, with the additionaly feature of *locating* the clusters using the Harvard-Oxford atlas.
(Of course, for this to be valid, you MUST work in MNI space!)
The location is an extra columns in the cluster table and contains the named of the regions in which your voxel belong, ordered by decreased probability.

(work in progress)

### Fancy feature: fmri concatenation (experimental!)

### Participant level (First-level)

### Group level (Second-level)

## The configuration files

Those are `json` files (so basically a text file with a special syntax that makes it easy to read and write both for humans and robots).



