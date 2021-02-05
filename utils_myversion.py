'''Editing their functions'''
import numpy as np

def compute_ap(gt_boxes, gt_class_ids, gt_masks,
           pred_boxes, pred_class_ids, pred_scores, pred_masks,
           iou_threshold=0):
    """Compute Average Precision at a set IoU threshold (default 0.5).
                                                                    
    Returns:
    mAP: Mean Average Precision
    precisions: List of precisions at different class score thresholds.
    recalls: List of recall values at different class score thresholds.
    overlaps: [pred_boxes, gt_boxes] IoU overlaps.
    """
    print('started computing ap')
    # Get matches and overlaps
    gt_match, pred_match, overlaps = compute_matches(
        gt_boxes, gt_class_ids, gt_masks,
        pred_boxes, pred_class_ids, pred_scores, pred_masks,
        iou_threshold)
    #print('already matched')

    # Compute precision and recall at each prediction box step
    precisions = np.cumsum(pred_match > -1) / (np.arange(len(pred_match)) + 1)
    recalls = np.cumsum(pred_match > -1).astype(np.float32) / len(gt_match)
    #print('computed precision')

    # Pad with start and end values to simplify the math
    precisions = np.concatenate([[0], precisions, [0]])
    recalls = np.concatenate([[0], recalls, [1]])
    #print('padded values')

    # Ensure precision values decrease but don't increase. This way, the
    # precision value at each recall threshold is the maximum it can be
    # for all following recall thresholds, as specified by the VOC paper.
    for i in range(len(precisions) - 2, -1, -1):
        precisions[i] = np.maximum(precisions[i], precisions[i + 1])

    # Compute mean AP over recall range
    indices = np.where(recalls[:-1] != recalls[1:])[0] + 1
    mAP = np.sum((recalls[indices] - recalls[indices - 1]) *
                 precisions[indices])
    print('finished returning ap')
    return mAP, precisions, recalls, overlaps


def compute_matches(gt_boxes, gt_class_ids, gt_masks,
                pred_boxes, pred_class_ids, pred_scores, pred_masks,
                iou_threshold=0.5, score_threshold=0.0):
    """Finds matches between prediction and ground truth instances.
                                                                                                                                 
    Returns:
        gt_match: 1-D array. For each GT box it has the index of the matched
                  predicted box.
        pred_match: 1-D array. For each predicted box, it has the index of
                    the matched ground truth box.
        overlaps: [pred_boxes, gt_boxes] IoU overlaps.
    """
    # Trim zero padding
    # TODO: cleaner to do zero unpadding upstream
    gt_boxes = trim_zeros(gt_boxes)
    gt_masks = gt_masks[..., :gt_boxes.shape[0]]
    pred_boxes = trim_zeros(pred_boxes)
    pred_scores = pred_scores[:pred_boxes.shape[0]]
    # Sort predictions by score from high to low
    indices = np.argsort(pred_scores)[::-1]
    pred_boxes = pred_boxes[indices]
    pred_class_ids = pred_class_ids[indices]
    pred_scores = pred_scores[indices]
    pred_masks = pred_masks[..., indices]

    #print('pred masks', pred_masks.shape)
    #print('gt_masks', gt_masks.shape)
    # Compute IoU overlaps [pred_masks, gt_masks]
    overlaps = compute_overlaps_masks(pred_masks, gt_masks)
    #print('found overlaps', overlaps)
    # Loop through predictions and find matching ground truth boxes
    match_count = 0
    pred_match = -1 * np.ones([pred_boxes.shape[0]])
    gt_match = -1 * np.ones([gt_boxes.shape[0]])
    #print('starting loop')
    for i in range(len(pred_boxes)):
       #print('i - starting loop', i)
       # Find best matching ground truth box
       # 1. Sort matches by score
       sorted_ixs = np.argsort(overlaps[i])[::-1]
       #print('sorted_ixs - sorting match by score', sorted_ixs)
       # 2. Remove low scores
       low_score_idx = np.where(overlaps[i, sorted_ixs] < score_threshold)[0]
       #print('low scoring matches', low_score_idx)
       if low_score_idx.size > 0:
           sorted_ixs = sorted_ixs[:low_score_idx[0]]
           #print('there was a low score, new sorted list', sorted_ixs)
       # 3. Find the match
       for j in sorted_ixs:
           #print('j in sorted_ixs', j)
           # If ground truth box is already matched, go to next one
           if gt_match[j] > -1:
               #print('j is already matched')
               continue
           # If we reach IoU smaller than the threshold, end the loop
           iou = overlaps[i, j]
           print('iou', iou)
           if iou < iou_threshold:
               #print('iou smaller than threshold')
               break
           # Do we have a match?
           if pred_class_ids[i] == gt_class_ids[j]:
               match_count += 1
               gt_match[j] = i
               pred_match[i] = j
               break
    print('match count', match_count)
    print('finished matching, here are the gt_match and pred_match values, which are indices', gt_match, pred_match)
    return gt_match, pred_match, overlaps


def compute_overlaps(boxes1, boxes2):
    """Computes IoU overlaps between two sets of boxes.
    boxes1, boxes2: [N, (y1, x1, y2, x2)].
    For better performance, pass the largest set first and the smaller second.
    """
    print('started computing overlaps')
    # Areas of anchors and GT boxes
    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])

    # Compute overlaps to generate matrix [boxes1 count, boxes2 count]
    # Each cell contains the IoU value.
    overlaps = np.zeros((boxes1.shape[0], boxes2.shape[0]))
    for i in range(overlaps.shape[1]):
        box2 = boxes2[i]
        overlaps[:, i] = compute_iou(box2, boxes1, area2[i], area1)
    print('finished computing overlaps')
    return overlaps
    
    
def trim_zeros(x):
    """It's common to have tensors larger than the available data and
    pad with zeros. This function removes rows that are all zeros.
                                                                                                                                 
    x: [rows, columns].
    """
    assert len(x.shape) == 2
    return x[~np.all(x == 0, axis=1)]

def compute_overlaps_masks(masks1, masks2):
    """Computes IoU overlaps between two sets of masks.
    masks1, masks2: [Height, Width, instances]
    
    Note added by  me: I was originally getting an error here because the recorded image size of my images was off, making the masks have different dimensions.
    """
    print('starting to compute overlaps')
    # If either set of masks is empty return empty result
    if masks1.shape[-1] == 0 or masks2.shape[-1] == 0:
        print('masks were empty')
        return np.zeros((masks1.shape[-1], masks2.shape[-1]))
    print('masks were not empty')
    # flatten masks and compute their areas
    masks1 = np.reshape(masks1 > .5, (-1, masks1.shape[-1])).astype(np.float32)
    masks2 = np.reshape(masks2 > .5, (-1, masks2.shape[-1])).astype(np.float32)
    area1 = np.sum(masks1, axis=0)
    area2 = np.sum(masks2, axis=0)
    print('flattened masks')
    # intersections and union
    intersections = np.dot(masks1.T, masks2)
    union = area1[:, None] + area2[None, :] - intersections
    overlaps = intersections / union
    print('finished computing overlaps')
    return overlaps
    
