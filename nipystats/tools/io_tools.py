"""
Various input/output tools
"""

def arguments_manager(version):
    """
        Wrapper to define and read arguments for main function call

    Args:
        version: version to output when calling with -h

    Returns:
        args as read from command line call
    """
    import argparse
    #  Deal with arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('bids_dir', help='The directory with the input '
                                         'dataset formatted according to '
                                         'the BIDS standard.')
    parser.add_argument('output_dir', help='The directory where the output '
                                           'files will be stored.')
    parser.add_argument('analysis_level', help='Level of the analysis that '
                                               'will be performed. '
                                               'Multiple participant level '
                                               'analyses can be run '
                                               'independently (in parallel)'
                                               ' using the same '
                                               'output_dir.',
                        choices=['participant', 'group'])
    parser.add_argument('--participant_label',
                        help='The label(s) of the participant(s) that should be analyzed. '
                             'The label corresponds to sub-<participant_label> from the BIDS spec '
                             '(so it does not include "sub-"). If this parameter is not provided all subjects '
                             'should be analyzed. Multiple participants can be specified with a space separated list.',
                        nargs="+")
    parser.add_argument('--task',
                        help='Task(s) to analyse (optional)',
                        nargs="+")
    parser.add_argument('--fmriprep_dir',
                        help='Path of the fmriprep derivatives. If ommited, set to bids_dir/derivatives/fmriprep')
    parser.add_argument('-v', '--version', action='version', version='nipystats version {}'.format(version))
    parser.add_argument('--config',
                        help='Path to json file fixing the pipeline parameters. '
                             'If omitted, default values will be used.')
    return parser.parse_args()

def get_fmriprep_dir(args):
    """
        Get and check existence of fmriprep dir from options or default

    Args:
        args: return from arguments_manager

    Returns:
        path to fmriprep dir
    """
    from os.path import join, isdir
    from .shell_tools import msg_error
    import sys
    #  fmriprep dir definition
    if args.fmriprep_dir:
        fmriprep_dir = args.fmriprep_dir
    else:
        fmriprep_dir = join(args.bids_dir, "derivatives", "fmriprep")
    # exists?
    if not isdir(fmriprep_dir):
        msg_error("fmriprep dir %s not found." % fmriprep_dir)
        sys.exit(1)

    return fmriprep_dir

def set_default_config(level):
    """
        Sets the default values for the pipeline configuration
    Args:
        level: str, 'participant' or 'group'

    Returns:
        dict, default config
    """

    config = {}

    if level == 'participant':

        config['model'] = ''
        config['trials_type'] = []
        config['contrasts'] = []
        config['tasks'] = []
        config['first_level_options'] = dict()
        config['concatenation_pairs'] = None

    if level == 'group':

        config['model'] = ''
        config['first_level_model'] = ''
        config['first_level_contrast'] = ''
        config['concat_tasks'] = None
        config['tasks'] = []
        config['covariates'] = []
        config['contrasts'] = []
        config['add_constant'] = False
        config['smoothing_fwhm'] = 0
        config['report_options'] = dict()
        config['paired'] = False
        config['task_weights'] = dict()

    return config

def read_config_file(file, level):
    """
        Read config file

    Args:
        file, path to json file
    Returns:
        dict, with various values/np.arrays for the parameters used in main script
    """

    config = set_default_config(level=level)

    import json
    from .shell_tools import msg_info

    msg_info('Reading parameters from user-provided configuration file %s' % file)
    with open(file, 'r') as f:
        config_json = json.load(f)

    if level == 'participant':

        keys = ['model', 'trials_type', 'contrasts',
                'tasks', 'first_level_options', 'concatenation_pairs']

        for key in keys:
            if key in config_json.keys():
                config[key] = config_json[key]

        if 'first_level_options' in config_json.keys():
            if 'signal_scaling' in config['first_level_options'].keys():
                config['first_level_options']['signal_scaling'] = tuple(config['first_level_options']['signal_scaling'])

    if level == 'group':

        keys = ['model', 'first_level_model', 'first_level_contrast',
                'concat_tasks', 'tasks', 'covariates', 'contrasts',
                'add_constant', 'smoothing_fwhm', 'report_options', 'paired', 'task_weights']

        for key in keys:
            if key in config_json.keys():
                config[key] = config_json[key]

    return config
