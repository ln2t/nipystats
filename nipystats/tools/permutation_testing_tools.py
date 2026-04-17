"""
Tools to perform permutation testing
"""

def apply_permutation_on_design_matrix(design_matrix, cols):

    # get columns NOT to be shuffled
    cols_to_keep = list(set(design_matrix.columns) - set(cols))

    # remove columns that must be preserved by shuffling
    sub_design_matrix = design_matrix[cols]

    # apply shuffling
    shuffled_sub_design_matrix = sub_design_matrix.sample(n=len(sub_design_matrix)).reset_index(drop=True)

    # re-add columns to keep
    shuffled_design_matrix = shuffled_sub_design_matrix.copy()
    shuffled_design_matrix[cols_to_keep] = design_matrix[cols_to_keep]

    return shuffled_design_matrix

