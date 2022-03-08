max_shape = 1280
import cv2
import numpy as np
from skimage.measure import ransac
from skimage.transform import ProjectiveTransform, AffineTransform

def resize(image, max_shape):
    if max(image.shape) >= max_shape:
        const = max_shape / max(image.shape)
        image = cv2.resize(image, (int(image.shape[1] * const), int(image.shape[0] * const)))
    return image

def if_normed(scale, rot):
    Bool = (0.3 < scale[0] < 8) and (0.3 < scale[1] < 8) and (max(scale[0], scale[1]) / min(scale[0], scale[1]) < 1.3) and (-np.pi / 8 < rot < np.pi / 4)
    return Bool

def ransac_match(src_pts,dst_pts, good_idx):

    model, inliers = ransac(
        (src_pts, dst_pts),
        AffineTransform, min_samples=4,
        residual_threshold=4, max_trials=20000
    )

    n_inliers = np.sum(inliers)
    matched_indices = [good_idx[idx] for idx in inliers.nonzero()[0]]  ###
    matches_amount = len(matched_indices)

    if n_inliers == 0:
        return None, [], 0

    return model, matched_indices, matches_amount

# https://github.com/amusi/Non-Maximum-Suppression/blob/master/nms.py
def nms(bounding_boxes, threshold):
    # If no bounding boxes, return empty list
    if len(bounding_boxes) == 0:
        return [], []

    # Bounding boxes
    boxes = np.array(bounding_boxes)
    confidence_score = np.array([0.9 for i in range(len(boxes))])

    # coordinates of bounding boxes
    start_x = boxes[:, 0]
    start_y = boxes[:, 1]
    end_x = boxes[:, 2]
    end_y = boxes[:, 3]

    # Confidence scores of bounding boxes
    score = np.array(confidence_score)

    # Picked bounding boxes
    picked_boxes = []
    picked_score = []

    # Compute areas of bounding boxes
    areas = (end_x - start_x + 1) * (end_y - start_y + 1)

    # Sort by confidence score of bounding boxes
    order = np.argsort(score)

    # Iterate bounding boxes
    while order.size > 0:
        # The index of largest confidence score
        index = order[-1]

        # Pick the bounding box with largest confidence score
        picked_boxes.append(bounding_boxes[index])
        picked_score.append(confidence_score[index])

        # Compute ordinates of intersection-over-union(IOU)
        x1 = np.maximum(start_x[index], start_x[order[:-1]])
        x2 = np.minimum(end_x[index], end_x[order[:-1]])
        y1 = np.maximum(start_y[index], start_y[order[:-1]])
        y2 = np.minimum(end_y[index], end_y[order[:-1]])

        # Compute areas of intersection-over-union
        w = np.maximum(0.0, x2 - x1 + 1)
        h = np.maximum(0.0, y2 - y1 + 1)
        intersection = w * h

        # Compute the ratio between intersection and union
        ratio = intersection / (areas[index] + areas[order[:-1]] - intersection)

        left = np.where(ratio < threshold)
        order = order[left]

    return picked_boxes


def predict_image(train_image, query_image):
    # https://stackoverflow.com/questions/42938149/opencv-feature-matching-multiple-objects
    # https://stackoverflow.com/questions/46607647/sift-feature-matching-point-coordinates

    sift = cv2.SIFT_create(nOctaveLayers=5)
    train_imag = cv2.cvtColor(train_image, cv2.COLOR_BGR2GRAY)
    query_image = cv2.cvtColor(query_image, cv2.COLOR_BGR2GRAY)
    train_image = resize(train_imag, max_shape)
    h, w = train_image.shape
    kp1, des1 = sift.detectAndCompute(train_image, None)
    kp2, des2 = sift.detectAndCompute(query_image, None)
#     x = np.array([kp2[0].pt])

#     for i in range(len(kp2)):
#         x = np.append(x, [kp2[i].pt], axis=0)

#     x = x[1:len(x)]
#     bandwidth = estimate_bandwidth(x, quantile=quantile, n_samples=n_samples)
#     ms = MeanShift(bandwidth=bandwidth, bin_seeding=True, cluster_all=True)
#     ms.fit(x)
#     labels = ms.labels_
#     labels_unique = np.unique(labels)
#     n_clusters_ = len(labels_unique)

#     s = [None] * n_clusters_
#     for i in range(n_clusters_):
#         l = ms.labels_
#         d, = np.where(l == i)
#         s[i] = list(kp2[xx] for xx in d)

#     query_coords = np.array([(0, 0), query_image.shape[::-1]]) #####
    bboxes = []
    FLANN_INDEX_KDTREE=0
    flannParam=dict(algorithm=FLANN_INDEX_KDTREE,tree=3)
    checks = dict(checks=50)
    flann=cv2.FlannBasedMatcher(flannParam,checks)
    des1 = np.float32(des1)
    des2 = np.float32(des2)
    k = 2
    while True:
        matches = flann.knnMatch(des1, des2, k )
        good = []
        good_idx = {}
        for i, (m, n) in enumerate(matches):
            if m.distance < 0.7 * n.distance:
                good.append(m)
                good_idx[len(good)-1] = i
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 2)
        model, matched_indices, matches_amount = ransac_match(src_pts,dst_pts,good_idx) #####
        # print("n_matched",matches_amount , "scale", model.scale)
        if matches_amount <= 4:
            break
#         model_scale, model_rotation = model.scale, model.rotation
        if if_normed(model.scale, model.rotation):
            h_query, w_query = query_image.shape[::-1]
            q_coordinates = np.array([(0, 0), (h_query, w_query)])
            coords = model.inverse(q_coordinates)
            bboxes.append((i, coords))
#         print(query_coords)
#         bbox_coords = model.inverse(query_coords)
#         bboxes.append((matches_amount,bbox_coords))
        kp1, des1 = [np.delete(x, matched_indices, axis=0) for x in [kp1, des1]]
#     else:
#         break
    bboxes = [b[1] for b in sorted(bboxes, key=lambda y: y[0], reverse=True)]
    try:
        bboxes = bbox2mns(bboxes)
        bboxes = nms(bboxes, 0.7)
        final_boxes = []
        for bbox in bboxes:
            final_boxes.append(tuple([bbox[0] / w, bbox[1] / h, (bbox[2] - bbox[0]) / w, (bbox[3] - bbox[0]) / h]))
    #         cv2.rectangle(train_image, (np.int16(bbox[0]), np.int16(bbox[1])),
#                                (np.int16(bbox[2]), np.int16(bbox[3])), (255, 0, 0), 8)
        return final_boxes
    except IndexError:
        return []

def bbox2mns(bboxes_input):
    bboxes = []
    for bbox in bboxes_input:
        bbox_tuple = (bbox[0][0],bbox[0][1],bbox[1][0],bbox[1][1])
        bboxes.append(bbox_tuple)
    return bboxes