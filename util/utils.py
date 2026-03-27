import numpy as np

def non_max_suppression(boxes, scores, threshold=0.5):
    if len(boxes) == 0:
        return []
    boxes = np.array(boxes)
    pick = []
    idxs = np.argsort(scores)

    while len(idxs) > 0:
        last = idxs[-1]
        pick.append(last)
        idxs = idxs[:-1]

    return pick