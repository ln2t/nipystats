from .shell_tools import *
from .permutation_testing_tools import apply_permutation_on_design_matrix

def harmonize_grid(img_list, ref_img):
    """
        Resample all imgs in img_list to ref_img. Return the list of resampled images.

    """

    from nilearn.image import resample_to_img

    output = []
    for i, img in enumerate(img_list):
        img_list[i] = resample_to_img(img, ref_img, interpolation='nearest')
    return img_list


def round_affine(img_list, n=2):
    """
        Round affines of each niimg in img_list. Do not chage in place, but returns the imgs with round affines.

    """

    from nilearn.image import load_img
    from nibabel import Nifti1Image
    import numpy as np
    output = []
    for i, m in enumerate(img_list):
        img = load_img(m)
        img_list[i] = Nifti1Image(img.get_fdata(), affine=np.round(img.affine, n), header=img.header)
        #_img = Nifti1Image(img.get_fdata(), affine=np.round(img.affine, n), header=img.header)
        #output.append(_img)

    return img_list


def bids_init(rawdata, output, fmriprep):
    from pathlib import Path
    from bids import BIDSLayout
    from os.path import join

    # create output dir
    Path(output).mkdir(parents=True, exist_ok=True)

    msg_info('Indexing BIDS dataset...')

    # bids init
    layout = BIDSLayout(rawdata)

    # add derivatives
    layout.add_derivatives(fmriprep)

    # save
    msg_info('Saving database at %s' % output)

    # initiate dataset_description file for outputs
    description = join(output, 'dataset_description.json')
    with open(description, 'w') as ds_desc:
        ds_desc.write('{"Name": \"%s\", "BIDSVersion": "v1.8.0", "DatasetType": "derivative", "GeneratedBy": [{"Name": \"%s\"}, {"Version": "dev"}]}' % ('nipystats', 'nipystats'))
        ds_desc.close()

    layout.add_derivatives(output)
    subjects = layout.get_subjects()
    tasks = layout.get_tasks()
    return layout, subjects, tasks


def camel_case(s, lower=False):
    """
        Converts string to camel case. If lower=True, use the lower camel case convention (first char is lower case).
        If lower=False, use Upper camel case (aka PascalCase).

    Args:
        s: string
        lower: bool, optional (default: False)

    Returns:
        camel case version of string

    """
    # Use regular expression substitution to replace underscores and hyphens with spaces,
    # then title case the string (capitalize the first letter of each word), and remove spaces
    from re import sub

    s = sub(r"(_|-)+", " ", s).title().replace(" ", "")

    if lower:
        # Join the string, ensuring the first letter is lowercase
        out_s = ''.join([s[0].lower(), s[1:]])
    else:
        out_s = s

    return out_s


def demean_fmri(img):
    from nilearn.image import load_img
    import numpy as np
    from nibabel import Nifti1Image
    if type(img) is str:
        img = load_img(img)
    original_data = img.get_fdata()
    data = np.zeros_like(original_data)

    nx, ny, nz, _ = img.shape

    mean = np.mean(original_data)

    for x in np.arange(nx):
        for y in np.arange(ny):
            for z in np.arange(nz):
                data[x, y, z, :] = original_data[x, y, z, :] - mean

    return Nifti1Image(data, header=img.header, affine=img.affine)

def concat_fmri(imgs, **kwargs):
    from nilearn import image

    # de-trend and high-pass

    if kwargs['high_pass'] is not None:
        msg_warning('High pass and t_r are assumed to be the same for the sessions to concatenate.')
        # todo: generalize with different t_r and high_pass values for each session to concatenate

    imgs = [image.clean_img(
                            i,
                            standardize=False,
                            detrend=True,
                            **kwargs
                            )
                            for i in imgs]

    # remove mean (clean_img with high_pass does not remove mean!)

    imgs = [demean_fmri(i) for i in imgs]

    return image.concat_imgs(imgs, auto_resample=True)


def make_block_design_matrix(dm1, dm2):
    import pandas as pd
    from scipy.linalg import block_diag
    import numpy as np
    dm12 = pd.DataFrame(block_diag(*[dm1, dm2]))
    dm12.columns = dm12.columns.astype("str")
    colnames = np.concatenate([dm1.columns.values, dm2.columns.values])
    dm12.columns.values[:] = colnames
    return dm12


def print_memory_usage():
    # Importing the library
    import psutil
    # Getting % usage of virtual_memory ( 3rd field)
    print('RAM memory % used:', psutil.virtual_memory()[2])
    # Getting usage of virtual_memory in GB ( 4th field)
    print('RAM Used (GB):', psutil.virtual_memory()[3]/1000000000)


def save_config(config, output_dir, label):
    import json
    from os.path import join
    from pathlib import Path
    label = camel_case(label)
    _output = join(output_dir, 'model-' + label)
    Path(_output).mkdir(exist_ok=True, parents=True)
    with open(join(_output, 'model-' + label + '_config.json'), 'w') as fp:
        json.dump(config, fp, indent=True)


class ParticipantAnalysis:
    """
        A class to perform first-level analysis. Mostly consists of wrappers for nilearn tools.

    """
    def __init__(self,
                 dataset_path=None,
                 subject=None,
                 task_label=None,
                 trials_type=None,
                 contrasts=None,
                 confound_strategy=None,
                 derivatives_folder=None,
                 model=None,
                 first_level_options=None,
                 report_options=None,
                 design_matrix=None,
                 bold_mask=None,
                 layout=None,
                 output_dir=None,
                 confounds=None,
                 model_name=None,
                 t_r=None,
                 high_pass=None):
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
        self.output_dir = output_dir
        self.confounds = confounds
        self.model_name = model_name
        self.t_r = t_r
        self.high_pass = high_pass

    def setup(self, dataset_path=None, task_label=None, subject=None, derivatives_folder=None, first_level_options=None):
        """
            Wrapper for nilearn's first_level_from_bids function.

        """
        from nilearn.glm.first_level import first_level_from_bids

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

        (models, imgs, all_events, all_confounds) = first_level_from_bids(dataset_path=dataset_path,
                                                                          task_label=task_label,
                                                                          sub_labels=[subject],
                                                                          derivatives_folder=derivatives_folder,
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
            msg_info('Only motion is allowed for confound_strategy')
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
            self.model = FirstLevelModel(mask_img=self.bold_mask, **self.first_level_options)
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

        if not self.contrasts is None:

            self.contrast_maps = {}
            for c in contrasts:
                self.contrast_maps[c] = self.model.compute_contrast(c, output_type = 'all')


    def plot_stat(self, contrast=None, threshold=5):
        """
            Wrapper for plot_stat_img

        """
        from nilearn.plotting import plot_stat_map

        sub = self.subject
        task = self.task_label

        if contrast is None:
            contrasts = self.contrasts
        else:
            contrasts = [contrast]

        if not self.contrasts is None:
            for c in contrasts:
                map = self.contrast_maps[c]['stat']
                plot_stat_map(map, threshold=threshold, title="%s, %s, %s (statistic)" % (c, sub, task))

    def generate_report(self, report_options=None):
        """
            Wrapper for FirstLevelModel.generate_report()

        """

        if report_options is None:
            report_options = {}

        if not self.contrasts is None:
            self.report = self.model.generate_report(self.contrasts,
                                                     **report_options)
            self.report_options = report_options
        else:
            msg_info('No contrast specified, thus no report to generate.')

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
        self.generate_report(report_options)

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

        if not self.contrasts is None:

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

        else:
            msg_info('No contrast specified, nothing to save.')

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


def get_group_confounds(layout, covariates=None):
    """
        Build confounds dataframe from BIDS participants.tsv file.
        Change name for nilearn compatibility and build numerical group confound.
        If 'group' is not in the coraviates, creates a 'constant' covariate
        (otherwise the constant term is a linear combination of the group covariates).

    """
    import pandas as pd
    rawdata_fn = layout.get(return_type='filename', extension='.tsv', scope='raw', subject=None, suffix='participants')[
        0]
    rawdata_df = pd.read_csv(rawdata_fn, sep='\t')

    if covariates is None:
        covariates = []

    # test if there is a multiplication in the covariates

    multiplicative_col_label = None

    for s in covariates:
        if '*' in s:
            covariates.remove(s)
            covs = s.split('*')
            if len(covs) == 2:
                pass
            else:
                msg_error('Covariates multiplication is supported only for two columns, ex. group*age')
            if 'group' in covs:
                covs.remove('group')
                multiplicative_col_label = covs[0]
                if multiplicative_col_label not in list(rawdata_df.columns.values):
                    msg_error('String %s is not part of the participants.tsv headers, cannot multiply.' % multiplicative_col_label)
                covariates.append('group')
            else:
                msg_error('Covariates multiplication is supported only for group and another column, ex. group*age')

    confounds = rawdata_df[['participant_id', *covariates]]
    confounds = confounds.rename(columns={'participant_id': 'subject_label'}).copy()

    if "group" in covariates:
        group_labels = set(confounds['group'].values)
        for _group in group_labels:
            confounds[_group] = 0
            confounds.loc[confounds['group'] == _group, _group] = 1

            if multiplicative_col_label is not None:
                confounds[_group] = confounds[_group] * rawdata_df[multiplicative_col_label]

        confounds.drop(columns=['group'], inplace=True)

    return confounds


def get_group_members_lists(layout, design_matrix_and_info):
    """
        Return a dict with keys = group_label and values = list of subject_label belonging to the group.
        Keeps only those subjects also present in design_matrix_and_info['subject_label'].

    """
    confounds = get_group_confounds(layout, covariates=['group'])

    group_members = dict()
    for col in confounds.columns:
        if not col == 'subject_label':
            group_members[col] = []
            for s in confounds.loc[confounds[col] == 1, 'subject_label'].values:
                if s in design_matrix_and_info['subject_label'].values:
                    group_members[col].append(camel_case(s))

    return group_members


def get_participantlevel_info(output_layout, tasks, model, contrast):
    """
        Make a dataframe with participant-level information from the outputs of nipystats at first level.

    """
    import pandas as pd
    df = pd.DataFrame()
    df['subject_label'] = []
    df['map_name'] = []

    if len(tasks) == 1:
        df['effects_map_path'] = []
    else:
        for t in tasks:
            df['effects_map_path' + '_' + t] = []

    if '+' in contrast:
        # this is a quick and dirty hack. Should fix this at first level by renaming Task1+Task2 by Task1PlusTark2 etc
        contrast = contrast.split('+')[0]

    for s in output_layout.get_subjects(**{'model': model, 'desc': contrast, 'task': tasks}):
        fns = []
        for t in tasks:
            results = output_layout.get(model=model, desc=contrast,
                               return_type='filename', extension='.nii.gz',
                               task=t, suffix='EffectSize', subject=s)
            if len(results)>0:
                fns.append(results[0])
        # todo: here we must solve the issue as for P22, for which just one task is missing. Do not add it, but then adapt for paired analysis. Also understand why I
        # got no error for the other models.
        if len(fns) == len(tasks):
            df.loc[len(df.index)] = ['sub-' + s, contrast, *fns]
        else:
            msg_warning('Subject %s does not seem to have all the tasks, not including it in the analysis.' % s)

    return df


def get_mask_intersect(layout, tasks):
    """
        Get all masks from layout.derivatives['fMRIPrep'] corresponding to task and computes intersection,
        taking care of affine and grid inconsistencies.

    """
    from nilearn.masking import intersect_masks

    if tasks is None:
        msg_warning('No tasks specified. Masks from anatomical scans might be grabbed in the process, which not only '
                    'is probably not what you want to do, but is also typically memory hungry.')

    masks = layout.derivatives['fMRIPrep'].get(desc='brain',
                                               return_type='filename',
                                               extension='.nii.gz',
                                               task=tasks,
                                               suffix='mask')
    ref_mask = masks[0]
    mask_imgs = round_affine(masks)
    mask_imgs = harmonize_grid(mask_imgs, ref_mask)
    return intersect_masks(mask_imgs), ref_mask


def niimg_is_nontrivial(img):
    """
        Check if img has at least one non-zero value
    Args:
        img: niimg

    Returns:
        bool
    """
    from nilearn.image import load_img
    import numpy as np
    data = load_img(img).get_fdata()
    return np.any(data)


def convert_neg_log10_pvals_to_z_threshold(neg_log10_pvals_img, z_map, alpha):
    from nilearn.image import math_img
    from nilearn.masking import apply_mask
    import numpy as np

    above_permutation_threshold_mask = math_img('img > -np.log10(%s)' % alpha, img=neg_log10_pvals_img)

    above_permutation_threshold_mask = round_affine([above_permutation_threshold_mask], 2)[0]

    if niimg_is_nontrivial(above_permutation_threshold_mask):
        return np.min(apply_mask(z_map, above_permutation_threshold_mask))
    else:
        return np.inf


class GroupAnalysis:
    def __init__(self,
                 layout=None,
                 tasks=None,
                 concat_tasks=None,
                 first_level_df=None,
                 first_level_model=None,
                 first_level_contrast=None,
                 confounds=None,
                 contrasts=None,
                 bold_mask=None,
                 model=None,
                 ref_mask=None,
                 design_matrix=None,
                 contrast_maps=None,
                 output_dir=None,
                 covariates=None,
                 glm=None,
                 add_constant=False,
                 paired=False,
                 task_weights=None,
                 smoothing_fwhm=8,
                 report_options=None,
                 contrasts_dict=None,
                 first_level_effect_maps=None,
                 false_postivive_control_strategies=None,
                 permutation_columns=None,
                 custom_z_threshold=None):
        self.layout = layout
        self.first_level_df = first_level_df
        self.first_level_model = camel_case(first_level_model)
        self.first_level_contrast = first_level_contrast
        self.contrasts = contrasts
        self.confounds = confounds
        self.bold_mask = bold_mask
        self.model = model
        self.ref_mask = ref_mask
        self.design_matrix = design_matrix
        self.contrast_maps = contrast_maps
        self.output_dir = output_dir
        self.covariates = covariates
        self.glm = glm
        self.tasks = tasks
        self.task = None
        self.concat_tasks = concat_tasks
        self.smoothing_fwhm = smoothing_fwhm
        self.report_options = report_options
        self.contrasts_dict = contrasts_dict
        self.first_level_effect_maps = first_level_effect_maps
        self.permutation_columns = permutation_columns
        self.custom_z_threshold = custom_z_threshold

        if false_postivive_control_strategies is None:
            # false_postivive_control_strategies = ['fpr', 'fdr', 'permutations', 'bonferroni']
            false_postivive_control_strategies = ['fpr']

        self.false_postivive_control_strategies = false_postivive_control_strategies

        if concat_tasks is None:
            self.n_tasks = len(tasks)

            if self.n_tasks == 1:
                self.task = tasks[0]
        else:
            self.n_tasks = 1
            self.task = concat_tasks

        self.add_constant = add_constant

        self.paired = paired
        if paired and task_weights is None:
            msg_info('Error: must provide task_weights for paired design')
        self.task_weights = task_weights

    def setup(self):
        """
            Basic setup steps to get things ready for business.
        """
        from nilearn.glm.second_level import SecondLevelModel
        from nilearn.image import load_img

        msg_info('Setup in progress...')
        # get first-level data paths for selected contrast
        self.output_layout = self.layout.derivatives['nipystats']
        msg_info("Number of subjects found in the outputs: %s" % str(len(self.output_layout.get_subjects())))
        msg_info("Computing intersect of all masks...")
        self.bold_mask, self.ref_mask = get_mask_intersect(self.layout, self.tasks)
        if self.concat_tasks is not None:
            self.tasks = [self.concat_tasks]
        self.glm = SecondLevelModel(mask_img=self.bold_mask, smoothing_fwhm=self.smoothing_fwhm)
        msg_info('Setup finished.')

    def make_design_matrix(self):
        """
            This is one of the most important function of the package, as it handles quite various cases for
            group-level designs.
        """
        msg_info('Making design matrix...')

        import pandas as pd

        dm = pd.DataFrame()
        tasks = self.tasks

        #if self.n_tasks == 1 or self.paired:
        if self.paired:
            # What follows take information from the first-level analysis together with the participants.tsv file
            # to build a design matrix, taking into account that maybe not all subjects has an output at first level
            # as well as group-level confounds (covariates)
            dm1 = get_participantlevel_info(self.output_layout, self.tasks, self.first_level_model,
                                            self.first_level_contrast).sort_values('subject_label')
            dm2 = get_group_confounds(layout=self.layout, covariates=self.covariates).sort_values('subject_label')
            dm2_filtered = dm2.loc[dm2['subject_label'].isin(dm1['subject_label'])].reset_index().drop(
                columns=['index'])
            dm = pd.concat([dm1, dm2_filtered.drop(columns=['subject_label'])], axis=1)
            dm.drop(columns=['map_name'], inplace=True)
        else:
            for t in self.tasks:
                # This is essentially the same as above, except that it creates at 'subject_task_label' into the
                # design matrix. This is to keep track of the various tasks that will be involved in the model
                # (without customization of the name, there would be ambiguities in the labels).
                dm1 = get_participantlevel_info(self.output_layout, [t], self.first_level_model,
                                                self.first_level_contrast).sort_values('subject_label')
                dm2 = get_group_confounds(layout=self.layout, covariates=self.covariates).sort_values('subject_label')
                dm2_filtered = dm2.loc[dm2['subject_label'].isin(dm1['subject_label'])].reset_index().drop(
                    columns=['index'])
                dm12 = pd.concat([dm1, dm2_filtered.drop(columns=['subject_label'])], axis=1)
                dm12['task'] = t
                # dm12['subject_task_label'] = dm12['subject_label'].apply(camel_case) + '_task-' + t
                dm = pd.concat([dm, dm12])

            if len(self.tasks) > 1:
                # create indicator function for task - this must be done after the loop above is completed
                for t in self.tasks:
                    dm[t] = 0
                    dm.loc[dm['task'] == t, t] = 1

            dm.rename(columns={0: 'subject_label'}, inplace=True)

            # if self.paired:
            #    # This handles the cases of paired analysis
            #    for t in tasks:
            #       if t not in self.task_weights.keys():
            #           msg_info('Error, task %s not in the task weights.' % t)

                # subjects = set(dm['subject_label'].values)

                # For paired designs, there must be one column per subject and per condition. Here we create such
                # cols, naming then 'sub-<s>_task-<task>'.
                # Moreover, the entries in the design matrix are set to what's given in task_weights
                # (typically, to compare two tasks, weights are something like [1, -1])

                # for s in subjects:
                #                     s = camel_case(s)
                #                     dm[s] = 0
                #                     for t in self.task_weights.keys():
                #                         dm.loc[dm['subject_task_label'] == s + '_task-' + t, s] = self.task_weights[t]


            # else:
                # Creates an indicator function for the tasks.
                #for t in tasks:

            dm.drop(columns=['map_name', 'task'], inplace=True)

        if self.add_constant:
            dm['constant'] = 1

            if self.covariates is not None and len(self.covariates) > 0:
                for cov in self.covariates:
                    # remove mean to numerical entries
                    import numpy as np
                    dm[cov] = dm[cov].values - np.mean(dm[cov].values)

        self.design_matrix_and_info = dm.copy()  # remaining non-numerical cols: ['subject_label', 'effects_map_path' + tasks]

        dm.drop(columns=['subject_label'], inplace=True)

        if self.paired:
            for t in tasks:
                dm.drop(columns=['effects_map_path' + '_' + t], inplace=True)
        else:
            dm.drop(columns=['effects_map_path'], inplace=True)

        self.design_matrix = dm

    def plot_design_matrix(self):
        from nilearn.plotting import plot_design_matrix
        plot_design_matrix(self.design_matrix)

    def get_first_level_effect_maps(self):
        """
           Prepare the data from the first level analysis for the second level step.
           Takes care of harmonizing the affines and the grids.

           For paired designs, compute the difference between the maps.
        """

        from nilearn.image import load_img, math_img

        df = self.design_matrix_and_info

        df.sort_values('subject_label')

        _list = []

        if self.paired:
            subjects = set(df['subject_label'].values)
            for s in subjects:

                t1 = self.tasks[0]
                t2 = self.tasks[1]

                img1 = df.loc[df['subject_label'] == s, 'effects_map_path' + '_' + t1].values[0]
                img2 = df.loc[df['subject_label'] == s, 'effects_map_path' + '_' + t2].values[0]

                [img1, img2] = round_affine([img1, img2])
                [img1, img2] = harmonize_grid([img1, img2], img1)

                c1 = self.task_weights[t1]
                c2 = self.task_weights[t2]

                _list.append(math_img('%s*img1 + %s*img2' % (c1, c2), img1=img1, img2=img2))
        else:
            _list = self.design_matrix_and_info['effects_map_path'].values

        _list = round_affine(_list)
        _list = list(harmonize_grid(_list, _list[0]))

        self.first_level_effect_maps = _list
        self.glm.target_affine = load_img(_list[0]).affine

    def fit(self):

        msg_info('Fitting model to data...')
        self.glm.fit(self.first_level_effect_maps, design_matrix=self.design_matrix)
        msg_info('Fit complete.')

    def compute_contrasts(self):

        msg_info('Computing contrasts...')

        # The contrasts_dict is useful to distinguish between the actual contrast string and the human-friendly name we
        # want to give him. The contrast string is typically complicated in paired designs (otherwise it is okay).
        # The nice name will be used for output names.
        contrasts_dict = {}

        for c in self.contrasts:
            contrasts_dict[c] = c

        # if self.paired:
        #     # For paired designs, the contrast strings are quite involved as they typically uses all cols
        #     # with 'sub-<s>_*'. Here we generate automatically such contrasts.
        #
        #     group_members = get_group_members_lists(self.layout, self.design_matrix_and_info)
        #
        #     msg_info('Contrasts automatically generated for paired analysis.')
        #
        #     for c in self.contrasts:
        #         contrasts_dict[c] = []
        #         if "+" in c:
        #             c_split = c.split("+")
        #             if len(c_split) == 2:
        #                 _str = "+".join(group_members[c_split[0]]) + "+" + "+".join(group_members[c_split[1]])
        #                 contrasts_dict[c] = _str
        #             else:
        #                 msg_info(
        #                     'Error in contrast definition, %s in not a valid value. More complicated contrasts not supported (yet...?)' % c)
        #         else:
        #             if "-" in c:
        #                 c_split = c.split("-")
        #                 if len(c_split) == 2:
        #                     _str = "+".join(group_members[c_split[0]]) + "-" + "-".join(group_members[c_split[1]])
        #                     contrasts_dict[c] = _str
        #                 else:
        #                     msg_info(
        #                         'Error in contrast definition, %s in not a valid value. More complicated contrasts not supported (yet...?)' % c)
        #             else:
        #                 _str = "+".join(group_members[c])
        #                 contrasts_dict[c] = _str
        #     self.contrasts = list(contrasts_dict.values())  # we need this for the report generation
        # else:
        #     for c in self.contrasts:
        #         contrasts_dict[c] = c

        # ... and finally compute the actual contrasts
        self.contrast_maps = {}
        for k in contrasts_dict.keys():
            self.contrast_maps[k] = self.glm.compute_contrast(second_level_contrast=contrasts_dict[k],
                                                              first_level_contrast=self.first_level_contrast,
                                                              output_type='all')

        self.contrasts_dict = contrasts_dict
        msg_info('Contrasts computed.')

    def compute_thresholds(self, n_perm=10, tfce=False):
        """
            Get thresholds for various false positive control strategies.
        """

        from nilearn.glm import threshold_stats_img

        self.thresholds = {}

        height_control_to_alpha = {}
        height_control_to_alpha['fpr'] = 0.001
        height_control_to_alpha['fdr'] = 0.05
        height_control_to_alpha['permutations'] = 0.05
        height_control_to_alpha['bonferroni'] = 0.05

        for k in self.contrasts_dict.keys():
            self.thresholds[k] = {}

            for strategy in self.false_postivive_control_strategies:

                if strategy == 'permutations':
                    #self.get_non_parametric_threshold(contrast=k,
                    #                                  alpha=height_control_to_alpha['permutations'],
                    #                                  n_perm=n_perm,
                    #                                  tfce=tfce)
                    self.run_permutation_testing(contrast=k, n_perm=n_perm)
                elif strategy == 'custom':
                    self.thresholds[k][strategy] = self.custom_z_threshold
                else:
                    _, self.thresholds[k][strategy] = threshold_stats_img(stat_img=self.contrast_maps[k]['z_score'],
                                                                          mask_img=self.bold_mask,
                                                                          alpha=height_control_to_alpha[strategy],
                                                                          height_control=strategy,
                                                                          two_sided=True)

    def generate_cluster_tables(self):
        """
            Wrapper for nilearn.reporting.get_cluster_table. Also finds
            location of clusters using Harvard-Oxford atlas (only for data in MNI space!).

        """

        from nilearn.reporting import get_clusters_table

        msg_info('Generating cluster table...')

        self.cluster_table = {}

        for k in self.contrasts_dict.keys():

            self.cluster_table[k] = {}

            for false_positive_control_strategy in self.false_postivive_control_strategies:

                _map = self.contrast_maps[k]['z_score']
                threshold = self.thresholds[k][false_positive_control_strategy]

                _df = get_clusters_table(_map, stat_threshold=threshold, two_sided=True)

                if _df.empty:
                    msg_info('No cluster for contrast %s for false positive control strategy %s' % (k,
                                                                                                    false_positive_control_strategy))
                else:
                    _df = get_location_harvard_oxford_on_df(_df)

                self.cluster_table[k][false_positive_control_strategy] = _df

    def generate_report(self):
        """
            Wrapper for nilearn.reporting.make_glm_report.

        """

        msg_info('Generating report(s)...')

        from nilearn.reporting import make_glm_report
        import numpy as np

        self.report = {}

        for hc in self.false_postivive_control_strategies:

            # for permutations, threshold is in principle contrast-dependent. To generate the report, we
            # should have different thresholds for each contrast. But the way reports works in nilearn do not
            # allow this. Instead, we choose to uniformly use the higher threshold, so that worst case
            # scenario we are too conservative. Of course, if all thresholds are equal, this also works.

            self.report[hc] = make_glm_report(model=self.glm,
                                              contrasts=self.contrasts,
                                              height_control=None,
                                              threshold=np.max([self.thresholds[k][hc] for k in self.contrasts_dict.keys()]),
                                              two_sided=True,
                                              cluster_threshold=10,
                                              plot_type='glass')

    def run_permutation_testing(self, contrast, alpha=0.05, n_perm=10):

        permuted_max_t_values = []
        for _ in range(n_perm):

            permuted_model = self.glm
            shuffled_design_matrix = apply_permutation_on_design_matrix(design_matrix=permuted_model.design_matrix_,
                                                                              cols=self.permutation_columns)
            permuted_model.fit(self.first_level_effect_maps, design_matrix=shuffled_design_matrix)

            maps = permuted_model.compute_contrast(second_level_contrast=contrast,
                                            first_level_contrast=self.first_level_contrast,
                                            output_type='all')
            permuted_max_t_values.append(get_max_in_niimg(maps['stat']))

        permuted_max_t_values

    def get_non_parametric_threshold(self, contrast, alpha=0.05, n_perm=10, tfce=False):

        msg_info("Starting permutations, this might take a while... (nperm = %s)" % n_perm)

        from nilearn.glm.second_level import non_parametric_inference
        npi = non_parametric_inference(self.first_level_effect_maps,
                                                       design_matrix=self.design_matrix,
                                                       second_level_contrast=self.contrasts_dict[contrast],
                                                       mask=self.bold_mask,
                                                       smoothing_fwhm=self.glm.smoothing_fwhm,
                                                       n_perm=n_perm,
                                                       tfce=tfce,
                                                       two_sided_test=True)

        if tfce:
            neg_log10_pvals_img = npi['logp_max_tfce']
        else:
            neg_log10_pvals_img = npi

        non_parametric_threshold = convert_neg_log10_pvals_to_z_threshold(neg_log10_pvals_img,
                                                                          self.contrast_maps[contrast]['z_score'],
                                                                          alpha)

        msg_info('Computed threshold from permutation testing (z-score): %s' % non_parametric_threshold)

        self.thresholds[contrast]["permutations"] = non_parametric_threshold

    def export_to_bids(self):

        msg_info('Exporting to BIDS...')

        from os.path import join
        from pathlib import Path

        pattern = "model-{model}/desc-{desc}/group/[{output_type}]/group[_task-{task}]_model-{model}[_desc-{desc}][_secondLevelModel-{secondLevelModel}][_secondLevelContrast-{secondLevelContrast}]_{suffix}{extension}"

        maps = self.contrast_maps

        self.maps_path = {}

        first_level_model = self.first_level_model
        first_level_contrast = self.first_level_contrast
        task = self.task

        output_layout = self.layout.derivatives['nipystats']
        output_dir = join(output_layout.root, 'model-' + camel_case(first_level_model), 'desc-' + first_level_contrast,
                          'group')
        self.output_dir = output_dir

        Path(output_dir).mkdir(exist_ok=True, parents=True)
        Path(join(output_dir, 'maps')).mkdir(exist_ok=True, parents=True)
        Path(join(output_dir, 'cluster_tables')).mkdir(exist_ok=True, parents=True)
        Path(join(output_dir, 'reports')).mkdir(exist_ok=True, parents=True)
        msg_info('Output directory is %s' % output_dir)

        for k in self.contrasts_dict.keys():
            self.maps_path[k] = {}
            for kk in maps[k].keys():
                entities = {'task': task, 'suffix': camel_case(kk),
                            'desc': first_level_contrast, 'model': camel_case(first_level_model),
                            'extension': '.nii.gz',
                            'secondLevelModel': camel_case(self.model),
                            'secondLevelContrast': camel_case(k),
                            'output_type': 'maps'}
                map_fn = output_layout.build_path(entities, pattern, validate=False)
                maps[k][kk].to_filename(map_fn)
                self.maps_path[k][kk] = map_fn

            for hc in self.false_postivive_control_strategies:
                # save also cluster tables with location in Harvard-Oxford atlas

                if type(hc) is dict:
                    hc = hc['Name']

                entities = {'task': task, 'suffix': camel_case('cluster table %s' % hc), 'desc': first_level_contrast,
                            'model': camel_case(first_level_model), 'secondLevelModel': camel_case(self.model),
                            'extension': '.csv', 'secondLevelContrast': camel_case(k), 'output_type': 'cluster_tables'}
                cluster_table_fn = output_layout.build_path(entities, pattern, validate=False)
                self.cluster_table[k][hc].to_csv(cluster_table_fn, sep=',')

        self.report_path = {}

        for hc in self.false_postivive_control_strategies:

            if type(hc) is dict:
                hc = hc['Name']

            entities = {'task': task, 'suffix': camel_case('report %s' % hc), 'desc': first_level_contrast,
                        'model': camel_case(first_level_model), 'secondLevelModel': camel_case(self.model),
                        'extension': '.html', 'output_type': 'reports'}
            rep_fn = output_layout.build_path(entities, pattern, validate=False)
            msg_info('Saving report for %s to %s' % (hc, rep_fn))
            self.report[hc].save_as_html(rep_fn)
            self.report_path[hc] = rep_fn

    def plot_stat_map(self, threshold=5):
        from nilearn.plotting import plot_stat_map
        contrasts_dict = self.contrasts_dict
        for k in contrasts_dict.keys():
            plot_stat_map(self.contrast_maps[k]['stat'], title='Group contrast %s, model %s, individual contrast %s' %
                                                               (k, self.first_level_model,
                                                                self.first_level_contrast), threshold=threshold)


def concat_ParticipantAnalyses(pa1, pa2):
    """
        Tool to concatenate two ParticipantAnalysis objects from two different tasks.
        The fMRI data are concatenated using another tool, concat_fmri.
        The rest of this function builds the concatenated design matrix from the individual design matrices,
        assembled in a block-diagonal matrix. Moreover,  the names of the individual regressors are adapted to reflect
        from which task they where taken from (to avoid duplicates); e.g. if there is a regressor called "motion
    Args:
        pa1: ParticipantAnalysis
        pa2: ParticipantAnalysis

    Returns:
        ParticipantAnalysis
    """

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
            _ll.append(_n + '_' + task1)
    dm1.columns.values[:] = _ll

    _ll = []
    for _n in dm2.columns.values:
        if _n == 'constant':
            _ll.append(task2)
        else:
            _ll.append(_n  + '_' + task2)
    dm2.columns.values[:] = _ll

    dm12 = make_block_design_matrix(dm1, dm2)

    pa12.design_matrix = dm12

    pa12.imgs = concat_fmri([pa1.imgs, pa2.imgs], high_pass=pa1.high_pass, t_r=pa1.t_r)

    pa12.dataset_path = pa1.dataset_path
    pa12.layout = pa1.layout

    return pa12


def check_tasks_for_concatenation_option(args, config):
    """
        Raise an error if a task argument was explicitly given
        while concatenation pairs where defined in config file
    Args:
        args: namespace
        config: dict

    Returns:
        None
    """
    import sys
    if config['concatenation_pairs'] is not None and args.task is not None:
        msg_error('Task argument cannot be used if concatenation pairs are defined in configuration file.')
        sys.exit(1)

    return None


def process_subject_and_task(layout, subject, task, config):
    """

    Args:
        layout: BIDSLayout
        subject: string
        task: string
        config: dict

    Returns:
        ParticipantAnalysis object
    """
    pa = ParticipantAnalysis()

    try:
        pa.setup(dataset_path=layout.root,
                 task_label=task,
                 subject=subject,
                 derivatives_folder=layout.derivatives['fMRIPrep'].root,
                 first_level_options= config['first_level_options'])
        pa.select_events(trials_type=config['trials_type'])
        pa.select_confounds(confound_strategy='motion')
        pa.fit()
        pa.compute_contrasts(config['contrasts'])
        pa.generate_report(config['report_options'])
        pa.bids_export(layout.derivatives['nipystats'].root, config['model'])
    except Exception as error:
        msg_error('There was an issue with %s, %s' % (subject, task))
        print(error)

    return pa


def get_repetition_time(layout, subject, task):
    """
        Find repetition time for subject and task, as read in json file.

    Args:
        layout: BIDSLayout
        subject: string
        task: string

    Returns:
        float
    """

    import json
    fn = layout.get(extension='json', scope='root', subject=subject, task=task)
    if len(fn) > 0:
        msg_error('Several json files found for %s, %s' % (subject, task))
    else:
        fn = fn[0]

    with open(fn, 'r') as f:
        info = json.load(f)

    return float(info['RepetitionTime'])


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
            msg_info('Processing %s, %s' % (s, t))
            # print_memory_usage()
            pa[t] = ParticipantAnalysis()

            try:
                pa[t].full_pipeline(dataset_path=rawdata, task_label=t,
                                 subject=s, derivatives_folder=fmriprep,
                                 trials_type=_trials_type, confound_strategy='motion',
                                contrasts=_contrasts, first_level_options=_first_level_options);
                pa[t].bids_export(output_dir, _model)
            except Exception as error:
                msg_error('There was an issue with %s, %s' % (s, t))
                print(error)

        if not _concat_pairs is None:
            msg_warning('using experimental concatenation tool.')
            for (t1, t2) in _concat_pairs:

                msg_info('Starting concatenation %s: %s + %s' % (s, t1, t2))
                # print_memory_usage()

                pa1 = pa[t1]
                pa2 = pa[t2]

                pa1.high_pass = config['high_pass']
                pa1.t_r = get_repetition_time(layout=layout, subject=s, task=t1)

                pa12 = concat_ParticipantAnalyses(pa1, pa2)
                pa12.fit()

                if _contrasts is None:
                    pa12.contrasts = [t1 + '+' + t2, t1 + '-' + t2]
                    pa12.compute_contrasts()
                    pa12.generate_report()
                    pa12.bids_export(output_dir, _model)
                else:
                    for c in _contrasts:
                        pa12.contrasts = [c + '_' + t1 + '+' + c + '_' + t2, t1 + '-' + t2, t1 + '+' + t2]
                        pa12.compute_contrasts()
                        pa12.generate_report()
                        pa12.bids_export(output_dir, _model)

                # pa[t1 + t2] = pa12
                del pa1, pa2, pa12

        del pa  # to save memory


def process_concatenation_of_tasks(layout, participant_analysis, config):
    """

    Args:
        layout: BIDSLayout
        participant_analysis: dict of ParticipantAnalysis objects
        pairs: list of strings
        contrasts: list of string

    Returns:
        ParticipantAnalysis object
    """

    pa = {}

    for (t1, t2) in config['concatenation_pairs']:

        msg_info('Starting concatenation %s + %s' % (t1, t2))
        print_memory_usage()

        pa1 = participant_analysis[t1]
        pa2 = participant_analysis[t2]

        pa[(t1, t2)] = concat_ParticipantAnalyses(pa1, pa2)
        print_memory_usage()

        pa[(t1, t2)].fit()

        if config['contrasts'] is None:
            pa[(t1, t2)].contrasts = [t1 + '+' + t2, t1 + '-' + t2]
        else:
            for c in config['contrasts']:
                pa[(t1, t2)].contrasts = [c + '_' + t1 + '+' + c + '_' + t2, t1 + '-' + t2, t1 + '+' + t2]

        pa[(t1, t2)].compute_contrasts()
        pa[(t1, t2)].generate_report()
        pa[(t1, t2)].bids_export(layout.derivatives['nipystats'].root, config['model'])

    return pa


def get_max_in_niimg(img):
    import numpy as np
    return np.nanmax(img.get_fdata())

def run_group_analysis_from_config(layout, config):

    model = config['model']
    first_level_model = config['first_level_model']
    first_level_contrast = config['first_level_contrast']
    concat_tasks = config['concat_tasks']
    tasks = config['tasks']
    covariates = config['covariates']
    contrasts = config['contrasts']
    add_constant = config['add_constant']
    smoothing_fwhm = config['smoothing_fwhm']
    paired = config['paired']
    task_weights = config['task_weights']
    permutation_columns = config['permutation_columns']
    custom_z_threshold = config['custom_z_threshold']

    ga = GroupAnalysis(layout=layout,
                       tasks=tasks,
                       first_level_model=first_level_model,
                       first_level_contrast=first_level_contrast,
                       model=model,
                       covariates=covariates,
                       contrasts=contrasts,
                       add_constant=add_constant,
                       smoothing_fwhm=smoothing_fwhm,
                       paired=paired,
                       concat_tasks=concat_tasks,
                       task_weights=task_weights,
                       false_postivive_control_strategies=['fpr', 'fdr', 'bonferroni', 'custom'],
                       permutation_columns=permutation_columns,
                       custom_z_threshold=custom_z_threshold)

    ga.setup()
    ga.make_design_matrix()
    ga.get_first_level_effect_maps()
    ga.fit()
    ga.compute_contrasts()
    ga.compute_thresholds()
    ga.generate_report()
    ga.generate_cluster_tables()
    ga.export_to_bids()


def mni_to_voxel(x, y, z, affine):
    import numpy as np
    mni = np.array([[x], [y], [z], [1]])
    voxel = np.linalg.inv(affine).dot(mni)
    return tuple(np.round(voxel[:3]).astype(int))


def read_xml(xml_file):
    import xml.etree.ElementTree as ET
    labels = {}
    tree = ET.parse(xml_file)
    root = tree.getroot()
    for label in root.findall('.//label'):
        index = int(label.get('index'))
        name = label.text.strip()
        labels[index] = name
    return labels


def get_location_harvard_oxford_on_df(df):

    import nibabel as nib
    import importlib.resources
    with importlib.resources.path('nipystats.data.atlases', 'HarvardOxford-Cortical.xml') as xml_file:
        with importlib.resources.path('nipystats.data.atlases', 'HarvardOxford-cort-prob-1mm.nii.gz') as nifti_file:
            # Load NIfTI file
            nifti = nib.load(nifti_file)
            data = nifti.get_fdata()
            # Get affine transformation matrix
            affine = nifti.affine

            df['Location (Harvard-Oxford)'] = df.apply(lambda row: get_location_harvard_oxford(row, data, affine, xml_file), axis=1)
    return df

def get_location_harvard_oxford(row, data, affine, xml_file):

    x = row['X']
    y = row['Y']
    z = row['Z']

    # Convert MNI coordinates to voxel coordinates
    voxel_x, voxel_y, voxel_z = mni_to_voxel(x, y, z, affine)

    # Get volumes at voxel coordinates
    volumes = data[voxel_x, voxel_y, voxel_z, :].flatten()

    # Read XML file to get label names
    labels = read_xml(xml_file)

    # Sort volumes by probability
    sorted_volumes = sorted(enumerate(volumes), key=lambda x: x[1], reverse=True)

    prob_threshold = 0.01

    # Print selected volumes with corresponding label names

    output = []

    for index, prob in sorted_volumes:
        label_name = labels.get(index, "Unknown")
        if prob > prob_threshold:
            _str = f"{label_name} ({prob} %)"
            output.append(_str)

    return ' and '.join(output)