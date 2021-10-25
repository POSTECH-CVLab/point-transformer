from . import functions


def assign_score_withk_halfkernel(score, point_input, knn_idx, aggregate='sum'):
    return functions.assign_score_withk_halfkernel(score, point_input, knn_idx, aggregate)


def assign_score_withk(score, point_input, center_input, knn_idx, aggregate='sum'):
    return functions.assign_score_withk(score, point_input, center_input, knn_idx, aggregate)


