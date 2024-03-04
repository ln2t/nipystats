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
    config = read_config_file(args.config)

    # print some summary before running
    # msg_info("Bids directory: %s" % layout.root)
    # msg_info("Fmriprep directory: %s" % fmriprep_dir)
    # msg_info("Subject(s) to analyse: %s" % subjects_to_analyze)
    # msg_info("Task to analyse: %s" % task)
    # msg_info("Selected space: %s" % space)

    if args.analysis_level == "participant":

        for subject_label in args.participant_label:
            msg_info("Running for participant %s" % subject_label)

            if args.task:
                tasks = args.task
            else:
                tasks = config['tasks']

            for task_label in tasks:
                config['subjects'] = [subject_label]
                config['tasks'] = [task_label]
                try:
                    run_analysis_from_config(args.bids_dir, args.output_dir, fmriprep_dir, config)
                except:
                    msg_error('There is an issue with subject %s, task %s' % (subject_label, task_label))

    # running group level
    elif args.analysis_level == "group":
        msg_info("Group analysis not implemented yet.")

    msg_info("The End!")

if __name__ == '__main__':
    main()
