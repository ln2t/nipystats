# nipystats
A python package for fMRI statistical analysis.

# TL;DR
This is basically a nilearn wrapper for first- and second-level analyzes. You need fMRIPrep outputs.

## Installation

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

```
nipystats /path/to/bids/rawdata /path/to/derivatives group --fmriprep_dir /path/to/fMRIPrep/dir --config /path/to/group_level-config.json
```
