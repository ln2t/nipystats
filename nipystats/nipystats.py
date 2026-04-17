#!/usr/bin/env python3
"""BIDS app to do standard first and second level analyzes on task-based fMRI data.
Mostly based on nilearn.
This app complies with the Brain Imaging Data Structure (BIDS) App
standard. It assumes the dataset is organised according to BIDS.
Intended to run in command line with arguments as described in the help
(callable with -h)

Author: Antonin Rovai

Created: Feb 2024
"""

from .tools import *
import sys

def main():
    """
        This is the main function and is called at the end of this script.

    Returns:
        None
    """
    __version__ = 'dev'
    args = arguments_manager(__version__)
    fmriprep_dir = get_fmriprep_dir(args)

    # msg_info("Indexing BIDS dataset...")

    # layout = get_bidslayout(args)
    # layout.add_derivatives(fmriprep_dir)
    # subjects_to_analyze = get_subjects_to_analyze(args, layout)
    # task = get_task(args, layout)
    # space, res = get_space(args, layout)
    # layout = setup_output_dir(args, __version__, layout)
    # output_dir = layout.derivatives['cvrmap'].root
        # print some summary before running
    # msg_info("Bids directory: %s" % layout.root)
    # msg_info("Fmriprep directory: %s" % fmriprep_dir)
    # msg_info("Subject(s) to analyse: %s" % subjects_to_analyze)
    # msg_info("Task to analyse: %s" % task)
    # msg_info("Selected space: %s" % space)

    config = read_config_file(args.config, args.analysis_level)
    layout, all_subjects, all_tasks = bids_init(args.bids_dir, args.output_dir, fmriprep_dir)

    if args.analysis_level == "participant":

        check_tasks_for_concatenation_option(args, config)
        if args.participant_label:
            subjects_to_process = args.participant_label
        else:
            subjects_to_process = all_subjects

        if config['tasks'] is None or config['tasks'] == 'All':
            tasks_to_process = all_tasks
        else:
            tasks_to_process = config['tasks']

        save_config(config, args.output_dir, config['model'])

        for s in subjects_to_process:
            msg_info("Running for participant %s" % s)

            participant_analysis = {}
            for t in tasks_to_process:
                msg_info('Processing %s, %s' % (s, t))
                participant_analysis[t] = process_subject_and_task(layout, s, t, config)

            if config['concatenation_pairs'] is not None:
                msg_warning('Using experimental concatenation tool.')
                process_concatenation_of_tasks(layout, participant_analysis, config)

    # running group level
    elif args.analysis_level == "group":

        run_group_analysis_from_config(layout, config)

    msg_info("The End!")

if __name__ == '__main__':
    main()
