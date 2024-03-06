from .shell_tools import *


def harmonize_grid(img_list, ref_img):
    """
        Resample all imgs in img_list to ref_img. Return the list of resampled images.

    """

    from nilearn.image import resample_to_img

    output = []
    for img in img_list:
        output.append(resample_to_img(img, ref_img, interpolation='nearest'))
    return output


def round_affine(img_list, n=2):
    """
        Round affines of each niimg in img_list. Do not chage in place, but returns the imgs with round affines.

    """

    from nilearn.image import load_img
    from nibabel import Nifti1Image
    import numpy as np
    output = []
    for m in img_list:
        img = load_img(m)
        _img = Nifti1Image(img.get_fdata(), affine=np.round(img.affine, n), header=img.header)
        output.append(_img)

    return output

def bids_init(rawdata, output, fmriprep):
    from pathlib import Path
    from bids import BIDSLayout
    from os.path import join
    # bids init
    layout = BIDSLayout(rawdata)

    # add derivatives
    layout.add_derivatives(fmriprep)

    # create output dir
    Path(output).mkdir(parents=True, exist_ok=True)

    # initiate dataset_description file for outputs
    description = join(output, 'dataset_description.json')
    with open(description, 'w') as ds_desc:
        ds_desc.write('{"Name": \"%s\", "BIDSVersion": "v1.8.0", "DatasetType": "derivative", "GeneratedBy": [{"Name": \"%s\"}, {"Version": "dev"}]}' % ('nipystats', 'nipystats'))
        ds_desc.close()

    layout.add_derivatives(output)
    subjects = layout.get_subjects()
    tasks = layout.get_tasks()
    return layout, subjects, tasks


# In[ ]:


from re import sub

# Define a function to convert a string to camel case
def camel_case(s):
    # Use regular expression substitution to replace underscores and hyphens with spaces,
    # then title case the string (capitalize the first letter of each word), and remove spaces
    s = sub(r"(_|-)+", " ", s).title().replace(" ", "")

    # Join the string, ensuring the first letter is lowercase
    return ''.join([s[0].lower(), s[1:]])


# In[ ]:


def concat_fmri(imgs):
    from nilearn import image
    clean_imgs = [image.clean_img(_i, standardize=True, detrend=True) for _i in imgs]

    return image.concat_imgs(clean_imgs, auto_resample=True)


# In[ ]:


def make_block_design_matrix(dm1, dm2):
    import pandas as pd
    from scipy.linalg import block_diag
    import numpy as np
    dm12 = pd.DataFrame(block_diag(*[dm1, dm2]))
    dm12.columns = dm12.columns.astype("str")
    colnames = np.concatenate([dm1.columns.values, dm2.columns.values])
    dm12.columns.values[:] = colnames
    return dm12


# In[ ]:


def print_memory_usage():
    # Importing the library
    import psutil
    # Getting % usage of virtual_memory ( 3rd field)
    print('RAM memory % used:', psutil.virtual_memory()[2])
    # Getting usage of virtual_memory in GB ( 4th field)
    print('RAM Used (GB):', psutil.virtual_memory()[3]/1000000000)


# In[ ]:


def save_config(config, output_dir, label):
    import json
    from os.path import join
    from pathlib import Path
    _output = join(output_dir, 'model-' + label)
    Path(_output).mkdir(exist_ok=True, parents=True)
    with open(join(_output, 'model-' + label + '_config.json'), 'w') as fp:
        json.dump(config, fp, indent=True)


# ### Class: ParticipantAnalysis

# In[ ]:


from nilearn.glm.first_level import first_level_from_bids

class ParticipantAnalysis:
    """
        A class to perform first-level analysis. Mostly consists of wrappers for nilearn tools.

    """
    def __init__(self, dataset_path=None, subject=None, task_label=None,
                 trials_type=None, contrasts=None, confound_strategy=None,
                 derivatives_folder=None, model=None, first_level_options=None,
                 report_options=None, design_matrix=None, bold_mask=None, layout=None):
        self.dataset_path = dataset_path
        self.subject = subject
        self.task_label = task_label
        self.trials_type = trials_type
        self.contrasts = contrasts
        self.confound_strategy = confound_strategy
        self.derivatives_folder = derivatives_folder
        self.model = model
        self.first_level_options = first_level_options
        self.is_fit_ = False
        self.report_options = report_options
        self.design_matrix = design_matrix
        self.bold_mask = bold_mask
        self.layout = layout

    def setup(self, dataset_path=None, task_label=None, subject=None, derivatives_folder=None, first_level_options=None):
        """
            Wrapper for nilearn's first_level_from_bids function.

        """
        if dataset_path is None:
            dataset_path = self.dataset_path
        else:
            self.dataset_path = dataset_path

        if task_label is None:
            task_label = self.task_label
        else:
            self.task_label = task_label

        if subject is None:
            subject = self.subject
        else:
            self.subject = subject

        if derivatives_folder is None:
            derivatives_folder = self.derivatives_folder
        else:
            self.derivatives_folder = derivatives_folder

        if first_level_options is None:
            first_level_options = self.first_level_options
        else:
            self.first_level_options = first_level_options

        (models, imgs, all_events, all_confounds) =  first_level_from_bids(dataset_path = dataset_path, task_label = task_label,
                              sub_labels = [subject], derivatives_folder = derivatives_folder,
                              **first_level_options)

        self.model = models[0]
        self.imgs = imgs[0][0]
        self.all_events = all_events[0][0]
        self.all_confounds = all_confounds[0][0]
        self.first_level_options = first_level_options

        self.load_bold_mask()
        self.model.mask_img = self.bold_mask

    def select_events(self, trials_type=None):
        """
            In the events.tsv file, maybe only some trial_type values must be included. This function filters for such types.

        """
        if trials_type is None:
            trials_type = self.trials_type
        else:
            self.trials_type = trials_type

        df = self.all_events
        self.trials_type = trials_type
        self.events = df[df['trial_type'].isin(trials_type)]

    def select_confounds(self, confound_strategy=None):
        """
            Selects the confounds from fMRIPrep confound file.

        """

        if confound_strategy is None:
            confound_strategy = self.confound_strategy
        else:
            self.confound_strategy = confound_strategy

        if confound_strategy == 'motion':
            motion_confounds = ['trans_x', 'trans_y', 'trans_z', 'rot_x', 'rot_y', 'rot_z']
            self.confounds = self.all_confounds[motion_confounds]
        else:
            print('Only motion is allowed for confound_strategy')
        self.confound_strategy = confound_strategy

    def fit(self):
        """
            Wrapper for FirstLevelModel.fit()

        """
        from nilearn.glm.first_level import FirstLevelModel

        if self.design_matrix is None:
            self.setup()
            self.model.fit(run_imgs=self.imgs, events=self.events, confounds=self.confounds)
            self.design_matrix = self.model.design_matrices_[0]
        else:
            if self.bold_mask is None:
                self.load_bold_mask()
            self.model = FirstLevelModel(mask_img = self.bold_mask, **self.first_level_options)
            self.model.fit(run_imgs=self.imgs, design_matrices=self.design_matrix)

        self.is_fit_ = True

    def compute_contrasts(self, contrasts=None):
        """
            Wrapper for FirstLevelModel.compute_contrast()

        """
        if contrasts is None:
            contrasts = self.contrasts
        else:
            self.contrasts = contrasts

        self.contrast_maps = {}
        for c in contrasts:
            self.contrast_maps[c] = self.model.compute_contrast(c, output_type = 'all')


    def plot_stat(self, contrast=None, threshold=5):
        """
            Wrapper for FirstLevelModel.fit()

        """
        from nilearn.plotting import plot_stat_map

        sub = self.subject
        task = self.task_label

        if contrast is None:
            contrasts = self.contrasts
        else:
            contrasts = [contrast]
        for c in contrasts:
            map = self.contrast_maps[c]['stat']
            plot_stat_map(map, threshold=threshold, title="%s, %s, %s (statistic)" % (c, sub, task))

    def generate_report(self, **report_options):
        """
            Wrapper for FirstLevelModel.generate_report()

        """
        self.report = self.model.generate_report(self.contrasts, **report_options)
        self.report_options = report_options


    def full_pipeline(self, dataset_path=None, subject=None, task_label=None, trials_type=None, contrasts=None, confound_strategy=None,
                 derivatives_folder=None, first_level_options=None, **report_options):
        """
            (almost)-All-in-one command to run the analysis

        """
        self.setup(dataset_path = dataset_path, task_label = task_label, subject = subject,
                   derivatives_folder = derivatives_folder, first_level_options = first_level_options)
        self.select_events(trials_type = trials_type)
        self.select_confounds(confound_strategy = confound_strategy)
        self.fit()
        self.compute_contrasts(contrasts)
        self.generate_report(**report_options)

    def bids_export(self, output_dir, model):
        """
            Save results as BIDS derivatives (contrast maps and report)

        """
        from pathlib import Path
        from os.path import join

        sub = self.subject
        task = self.task_label

        patterns = {}

        patterns['map']    = "model-{model}/desc-{desc}/sub-{subject}/sub-{subject}_task-{task}_model-{model}_desc-{desc}_{suffix}.nii.gz"
        patterns['report'] = "model-{model}/sub-{subject}_task-{task}_model-{model}_{suffix}.html"

        if self.layout is None:
            self.load_layout()

        description = join(output_dir, 'dataset_description.json')

        with open(description, 'w') as ds_desc:
            ds_desc.write('{"Name": \"%s\", "BIDSVersion": "v1.8.0", "DatasetType": "derivative", "GeneratedBy": [{"Name": \"%s\"}, {"Version": "dev"}]}' % ('nipystats', 'nipystats'))
            ds_desc.close()

        if not any(['nipystats' in k for k in self.layout.derivatives.keys()]):
            self.layout.add_derivatives(output_dir)

        output_layout = self.layout.derivatives['nipystats']

        maps = self.contrast_maps

        self.maps_path = {}

        for c in self.contrasts:
            Path(join(output_layout.root, 'model-' + camel_case(model), 'desc-' + camel_case(c), 'sub-' + sub)).mkdir(parents=True, exist_ok=True)
            self.maps_path[c] = {}
            for k in maps[c].keys():
                entities = {'subject': sub, 'task': task, 'model': camel_case(model),
                            'desc': camel_case(c), 'suffix': camel_case(k)}
                map_fn = output_layout.build_path(entities, patterns['map'], validate=False)
                maps[c][k].to_filename(map_fn)
                self.maps_path[c][k] = map_fn

        entities = {'subject': sub, 'task': task, 'model': camel_case(model), 'suffix': 'report'}
        rep_fn = output_layout.build_path(entities, patterns['report'], validate=False)
        self.report.save_as_html(rep_fn)
        self.report_path = rep_fn

    def load_layout(self):
        """
            Wrapper for BIDSLayout

        """
        from bids import BIDSLayout
        self.layout = BIDSLayout(self.dataset_path)
        self.layout.add_derivatives(self.derivatives_folder)

    def load_bold_mask(self):
        """
            Load BOLD mask as given by fMRIPrep

        """
        subject = self.subject
        task = self.task_label
        layout = self.layout
        if self.layout is None:
            self.load_layout()
        self.bold_mask = self.layout.derivatives['fMRIPrep'].get(return_type = 'filename', subject=subject, task=task,
                                                  suffix='mask', desc='brain', extension='.nii.gz')[0]

    def plot_carpet(self):
        """
            Wrapper for nilearn's plot_carpet()

        """
        from nilearn.plotting import plot_carpet

        _ = plot_carpet(self.imgs, self.bold_mask, title="Data for %s, %s" % (self.subject, self.task_label))

    def plot_design_matrix(self):
        """
            Wrapper for nilearn's plot_design_matrix()

        """
        from nilearn.plotting import plot_design_matrix

        plot_design_matrix(self.model.design_matrices_[0])

    def plot_glass_brain(self, threshold=5):
        """
            Wrapper for nilearn's plot_glass_brain()

        """
        from nilearn.plotting import plot_glass_brain

        sub = self.subject
        task = self.task_label
        contrasts = self.contrasts
        for c in contrasts:
            map = self.contrast_maps[c]['stat']
            plot_glass_brain(map, threshold=threshold,
                             title="%s, %s, %s (statistic)" % (c, sub, task),
                             colorbar=True, vmax=10, cmap='cold_hot', plot_abs=False)

    def __del__(self):
        pass


# ### Function: concat_ParticipantAnalyses

# In[ ]:


def concat_ParticipantAnalyses(pa1, pa2):

    from nilearn.masking import intersect_masks

    pa12 = ParticipantAnalysis()

    if pa1.subject == pa2.subject:
        pa12.subject = pa1.subject

    pa12.task_label = pa1.task_label + pa2.task_label

    if not pa1.is_fit_:
        pa1.setup()
        pa1.select_events()
        pa1.select_confounds()
        pa1.fit()
    if not pa2.is_fit_:
        pa2.setup()
        pa2.select_events()
        pa2.select_confounds()
        pa2.fit()

    pa12.first_level_options = pa1.first_level_options
    pa12.report_options = pa1.report_options

    pa1.load_bold_mask()
    pa2.load_bold_mask()

    mask1 = pa1.bold_mask
    mask2 = pa2.bold_mask

    rounded_ = round_affine([mask1, mask2])
    _list = harmonize_grid(rounded_, rounded_[0])
    mask12 = intersect_masks(_list)

    pa12.bold_mask = mask12

    dm1 = pa1.model.design_matrices_[0].copy(deep=True)
    dm2 = pa2.model.design_matrices_[0].copy(deep=True)

    task1 = pa1.task_label
    task2 = pa2.task_label

    _ll = []
    for _n in dm1.columns.values:
        if _n == 'constant':
            _ll.append(task1)
        else:
            _ll.append(_n  + '_' + task1)
    dm1.columns.values[:] = _ll

    _ll = []
    for _n in dm2.columns.values:
        if _n == 'constant':
            _ll.append(task2)
        else:
            _ll.append(_n  + '_' + task2)
    dm2.columns.values[:] = _ll

    # dm1.columns.values[dm1.columns.values == 'constant'] = [task1]
    # dm2.columns.values[dm2.columns.values == 'constant'] = [task2]

    dm12 = make_block_design_matrix(dm1, dm2)

    pa12.design_matrix = dm12

    pa12.imgs = concat_fmri([pa1.imgs, pa2.imgs])

    pa12.dataset_path = pa1.dataset_path
    pa12.layout = pa1.layout

    return pa12

def run_analysis_from_config(rawdata, output_dir, subjects, fmriprep, config):

    _model = config['model']
    _trials_type = config['trials_type']
    _contrasts = config['contrasts']
    _tasks = config['tasks']
    _first_level_options = config['first_level_options']
    _concat_pairs = config['concatenation_pairs']

    layout, all_subjects, all_tasks = bids_init(rawdata, output_dir, fmriprep)

    if subjects:
        _subjects = subjects
    else:
        _subjects = all_subjects

    save_config(config, output_dir, _model)

    if _tasks is None or _tasks == 'All':
        _tasks = all_tasks

    for s in _subjects:
        msg_info("Running for participant %s" % s)
        pa = {}
        for t in _tasks:
            print('Processing %s, %s' % (s, t))
            print_memory_usage()
            pa[t] = ParticipantAnalysis()
            try:
                pa[t].full_pipeline(dataset_path=rawdata, task_label=t,
                                 subject=s, derivatives_folder=fmriprep,
                                 trials_type=_trials_type, confound_strategy='motion',
                                contrasts=_contrasts, first_level_options=_first_level_options);
                pa[t].bids_export(output_dir, _model)
            except:
                msg_error('There was an issue with %s, %s' % (s, t))

        if not _concat_pairs is None:
            print('Warning: using experimental concatenation tool.')
            for (t1, t2) in _concat_pairs:

                print('Starting concatenation %s: %s + %s' % (s, t1, t2))
                print_memory_usage()

                pa1 = pa[t1]
                pa2 = pa[t2]

                pa12 = concat_ParticipantAnalyses(pa1, pa2)
                pa12.fit()

                for c in _contrasts:
                    pa12.contrasts = [c + '_' + t1 + '+' + c + '_' + t2, t1 + '-' + t2, t1 + '+' + t2]
                    pa12.compute_contrasts()
                    pa12.generate_report()
                    pa12.bids_export(output_dir, _model)

                # pa[t1 + t2] = pa12
                del pa1, pa2, pa12

        del pa  # to save memory



