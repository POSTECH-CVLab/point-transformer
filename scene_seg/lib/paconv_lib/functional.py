from . import functions


def assign_score_withk(score, feat, center_feat, grouped_idx, aggregate='sum'):
    return functions.assign_score_withk(score, feat, center_feat, grouped_idx, aggregate)


