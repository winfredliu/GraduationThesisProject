from functools import partial

import cv2
import numpy as np
import random
from dataloaders import functions as F

# 增加数据预处理
interp_dict = {
    'NEAREST': cv2.INTER_NEAREST,
    'LINEAR': cv2.INTER_LINEAR,
    'CUBIC': cv2.INTER_CUBIC,
    'AREA': cv2.INTER_AREA,
    'LANCZOS4': cv2.INTER_LANCZOS4
}

class Resize(object):
    """
    Resize input.
    - If `target_size` is an int, resize the image(s) to (`target_size`, `target_size`).
    - If `target_size` is a list or tuple, resize the image(s) to `target_size`.
    Attention: If `interp` is 'RANDOM', the interpolation method will be chosen randomly.
    Args:
        target_size (int | list[int] | tuple[int]): Target size. If it is an integer, the
            target height and width will be both set to `target_size`. Otherwise,
            `target_size` represents [target height, target width].
        interp (str, optional): Interpolation method for resizing image(s). One of
            {'NEAREST', 'LINEAR', 'CUBIC', 'AREA', 'LANCZOS4', 'RANDOM'}.
            Defaults to 'LINEAR'.
        keep_ratio (bool, optional): If True, the scaling factor of width and height will
            be set to same value, and height/width of the resized image will be not
            greater than the target width/height. Defaults to False.
    Raises:
        TypeError: Invalid type of target_size.
        ValueError: Invalid interpolation method.
    """

    def __init__(self, target_size, interp='LINEAR', keep_ratio=False):
        super(Resize, self).__init__()
        if not (interp == "RANDOM" or interp in interp_dict):
            raise ValueError("`interp` should be one of {}.".format(
                interp_dict.keys()))
        if isinstance(target_size, int):
            target_size = (target_size, target_size)
        else:
            if not (isinstance(target_size,
                               (list, tuple)) and len(target_size) == 2):
                raise TypeError(
                    "`target_size` should be an int or a list of length 2, but received {}.".
                    format(target_size))
        # (height, width)
        self.target_size = target_size
        self.interp = interp
        self.keep_ratio = keep_ratio

    def apply_im(self, image, interp, target_size):
        flag = image.shape[2] == 1
        image = cv2.resize(image, target_size, interpolation=interp)
        if flag:
            image = image[:, :, np.newaxis]
        return image

    def apply_mask(self, mask, target_size):
        mask = cv2.resize(mask, target_size, interpolation=cv2.INTER_NEAREST)
        return mask

    def apply_bbox(self, bbox, scale, target_size):
        im_scale_x, im_scale_y = scale
        bbox[:, 0::2] *= im_scale_x
        bbox[:, 1::2] *= im_scale_y
        bbox[:, 0::2] = np.clip(bbox[:, 0::2], 0, target_size[0])
        bbox[:, 1::2] = np.clip(bbox[:, 1::2], 0, target_size[1])
        return bbox

    def apply_segm(self, segms, im_size, scale):
        im_h, im_w = im_size
        im_scale_x, im_scale_y = scale
        resized_segms = []
        for segm in segms:
            if F.is_poly(segm):
                # Polygon format
                resized_segms.append([
                    F.resize_poly(poly, im_scale_x, im_scale_y) for poly in segm
                ])
            else:
                # RLE format
                resized_segms.append(
                    F.resize_rle(segm, im_h, im_w, im_scale_x, im_scale_y))

        return resized_segms

    def apply(self, sample):
        sample['trans_info'].append(('resize', sample['image'].shape[0:2]))
        if self.interp == "RANDOM":
            interp = random.choice(list(interp_dict.values()))
        else:
            interp = interp_dict[self.interp]
        im_h, im_w = sample['image'].shape[:2]

        im_scale_y = self.target_size[0] / im_h
        im_scale_x = self.target_size[1] / im_w
        target_size = (self.target_size[1], self.target_size[0])
        if self.keep_ratio:
            scale = min(im_scale_y, im_scale_x)
            target_w = int(round(im_w * scale))
            target_h = int(round(im_h * scale))
            target_size = (target_w, target_h)
            im_scale_y = target_h / im_h
            im_scale_x = target_w / im_w

        sample['image'] = self.apply_im(sample['image'], interp, target_size)
        if 'image2' in sample:
            sample['image2'] = self.apply_im(sample['image2'], interp,
                                             target_size)

        if 'mask' in sample:
            sample['mask'] = self.apply_mask(sample['mask'], target_size)
        if 'aux_masks' in sample:
            sample['aux_masks'] = list(
                map(partial(
                    self.apply_mask, target_size=target_size),
                    sample['aux_masks']))
        if 'gt_bbox' in sample and len(sample['gt_bbox']) > 0:
            sample['gt_bbox'] = self.apply_bbox(
                sample['gt_bbox'], [im_scale_x, im_scale_y], target_size)
        if 'gt_poly' in sample and len(sample['gt_poly']) > 0:
            sample['gt_poly'] = self.apply_segm(
                sample['gt_poly'], [im_h, im_w], [im_scale_x, im_scale_y])
        if 'target' in sample:
            if 'sr_factor' in sample:
                # For SR tasks
                sample['target'] = self.apply_im(
                    sample['target'], interp,
                    F.calc_hr_shape(target_size, sample['sr_factor']))
            else:
                # For non-SR tasks
                sample['target'] = self.apply_im(sample['target'], interp,
                                                 target_size)

        sample['im_shape'] = np.asarray(
            sample['image'].shape[:2], dtype=np.float32)
        if 'scale_factor' in sample:
            scale_factor = sample['scale_factor']
            sample['scale_factor'] = np.asarray(
                [scale_factor[0] * im_scale_y, scale_factor[1] * im_scale_x],
                dtype=np.float32)
        return sample

class RandomCrop(object):
    """
    Randomly crop the input.
    1. Compute the height and width of cropped area according to `aspect_ratio` and
        `scaling`.
    2. Locate the upper left corner of cropped area randomly.
    3. Crop the image(s).
    4. Resize the cropped area to `crop_size` x `crop_size`.
    Args:
        crop_size (int | list[int] | tuple[int]): Target size of the cropped area. If
            None, the cropped area will not be resized. Defaults to None.
        aspect_ratio (list[float], optional): Aspect ratio of cropped region in
            [min, max] format. Defaults to [.5, 2.].
        thresholds (list[float], optional): Iou thresholds to decide a valid bbox
            crop. Defaults to [.0, .1, .3, .5, .7, .9].
        scaling (list[float], optional): Ratio between the cropped region and the
            original image in [min, max] format. Defaults to [.3, 1.].
        num_attempts (int, optional): Max number of tries before giving up.
            Defaults to 50.
        allow_no_crop (bool, optional): Whether returning without doing crop is
            allowed. Defaults to True.
        cover_all_box (bool, optional): Whether to ensure all bboxes be covered in
            the final crop. Defaults to False.
    """

    def __init__(self,
                 crop_size=None,
                 aspect_ratio=[.5, 2.],
                 thresholds=[.0, .1, .3, .5, .7, .9],
                 scaling=[.3, 1.],
                 num_attempts=50,
                 allow_no_crop=True,
                 cover_all_box=False):
        super(RandomCrop, self).__init__()
        self.crop_size = crop_size
        self.aspect_ratio = aspect_ratio
        self.thresholds = thresholds
        self.scaling = scaling
        self.num_attempts = num_attempts
        self.allow_no_crop = allow_no_crop
        self.cover_all_box = cover_all_box

    def _generate_crop_info(self, sample):
        im_h, im_w = sample['image'].shape[:2]
        if 'gt_bbox' in sample and len(sample['gt_bbox']) > 0:
            thresholds = self.thresholds
            if self.allow_no_crop:
                thresholds.append('no_crop')
            np.random.shuffle(thresholds)
            for thresh in thresholds:
                if thresh == 'no_crop':
                    return None
                for i in range(self.num_attempts):
                    crop_box = self._get_crop_box(im_h, im_w)
                    if crop_box is None:
                        continue
                    iou = self._iou_matrix(
                        sample['gt_bbox'],
                        np.array(
                            [crop_box], dtype=np.float32))
                    if iou.max() < thresh:
                        continue
                    if self.cover_all_box and iou.min() < thresh:
                        continue
                    cropped_box, valid_ids = self._crop_box_with_center_constraint(
                        sample['gt_bbox'], np.array(
                            crop_box, dtype=np.float32))
                    if valid_ids.size > 0:
                        return crop_box, cropped_box, valid_ids
        else:
            for i in range(self.num_attempts):
                crop_box = self._get_crop_box(im_h, im_w)
                if crop_box is None:
                    continue
                return crop_box, None, None
        return None

    def _get_crop_box(self, im_h, im_w):
        scale = np.random.uniform(*self.scaling)
        if self.aspect_ratio is not None:
            min_ar, max_ar = self.aspect_ratio
            aspect_ratio = np.random.uniform(
                max(min_ar, scale**2), min(max_ar, scale**-2))
            h_scale = scale / np.sqrt(aspect_ratio)
            w_scale = scale * np.sqrt(aspect_ratio)
        else:
            h_scale = np.random.uniform(*self.scaling)
            w_scale = np.random.uniform(*self.scaling)
        crop_h = im_h * h_scale
        crop_w = im_w * w_scale
        if self.aspect_ratio is None:
            if crop_h / crop_w < 0.5 or crop_h / crop_w > 2.0:
                return None
        crop_h = int(crop_h)
        crop_w = int(crop_w)
        crop_y = np.random.randint(0, im_h - crop_h)
        crop_x = np.random.randint(0, im_w - crop_w)
        return [crop_x, crop_y, crop_x + crop_w, crop_y + crop_h]

    def _iou_matrix(self, a, b):
        tl_i = np.maximum(a[:, np.newaxis, :2], b[:, :2])
        br_i = np.minimum(a[:, np.newaxis, 2:], b[:, 2:])

        area_i = np.prod(br_i - tl_i, axis=2) * (tl_i < br_i).all(axis=2)
        area_a = np.prod(a[:, 2:] - a[:, :2], axis=1)
        area_b = np.prod(b[:, 2:] - b[:, :2], axis=1)
        area_o = (area_a[:, np.newaxis] + area_b - area_i)
        return area_i / (area_o + 1e-10)

    def _crop_box_with_center_constraint(self, box, crop):
        cropped_box = box.copy()

        cropped_box[:, :2] = np.maximum(box[:, :2], crop[:2])
        cropped_box[:, 2:] = np.minimum(box[:, 2:], crop[2:])
        cropped_box[:, :2] -= crop[:2]
        cropped_box[:, 2:] -= crop[:2]

        centers = (box[:, :2] + box[:, 2:]) / 2
        valid = np.logical_and(crop[:2] <= centers,
                               centers < crop[2:]).all(axis=1)
        valid = np.logical_and(
            valid, (cropped_box[:, :2] < cropped_box[:, 2:]).all(axis=1))

        return cropped_box, np.where(valid)[0]

    def _crop_segm(self, segms, valid_ids, crop, height, width):
        crop_segms = []
        for id in valid_ids:
            segm = segms[id]
            if F.is_poly(segm):
                # Polygon format
                crop_segms.append(F.crop_poly(segm, crop))
            else:
                # RLE format
                crop_segms.append(F.crop_rle(segm, crop, height, width))

        return crop_segms

    def apply_im(self, image, crop):
        x1, y1, x2, y2 = crop
        return image[y1:y2, x1:x2, :]

    def apply_mask(self, mask, crop):
        x1, y1, x2, y2 = crop
        return mask[y1:y2, x1:x2, ...]

    def apply(self, sample):
        crop_info = self._generate_crop_info(sample)
        if crop_info is not None:
            crop_box, cropped_box, valid_ids = crop_info
            im_h, im_w = sample['image'].shape[:2]
            sample['image'] = self.apply_im(sample['image'], crop_box)
            if 'image2' in sample:
                sample['image2'] = self.apply_im(sample['image2'], crop_box)
            if 'gt_poly' in sample and len(sample['gt_poly']) > 0:
                crop_polys = self._crop_segm(
                    sample['gt_poly'],
                    valid_ids,
                    np.array(
                        crop_box, dtype=np.int64),
                    im_h,
                    im_w)
                if [] in crop_polys:
                    delete_id = list()
                    valid_polys = list()
                    for idx, poly in enumerate(crop_polys):
                        if not poly:
                            delete_id.append(idx)
                        else:
                            valid_polys.append(poly)
                    valid_ids = np.delete(valid_ids, delete_id)
                    if not valid_polys:
                        return sample
                    sample['gt_poly'] = valid_polys
                else:
                    sample['gt_poly'] = crop_polys

            if 'gt_bbox' in sample and len(sample['gt_bbox']) > 0:
                sample['gt_bbox'] = np.take(cropped_box, valid_ids, axis=0)
                sample['gt_class'] = np.take(
                    sample['gt_class'], valid_ids, axis=0)
                if 'gt_score' in sample:
                    sample['gt_score'] = np.take(
                        sample['gt_score'], valid_ids, axis=0)
                if 'is_crowd' in sample:
                    sample['is_crowd'] = np.take(
                        sample['is_crowd'], valid_ids, axis=0)

            if 'mask' in sample:
                sample['mask'] = self.apply_mask(sample['mask'], crop_box)

            if 'aux_masks' in sample:
                sample['aux_masks'] = list(
                    map(partial(
                        self.apply_mask, crop=crop_box),
                        sample['aux_masks']))

            if 'target' in sample:
                if 'sr_factor' in sample:
                    sample['target'] = self.apply_im(
                        sample['target'],
                        F.calc_hr_shape(crop_box, sample['sr_factor']))
                else:
                    sample['target'] = self.apply_im(sample['image'], crop_box)

        if self.crop_size is not None:
            sample = Resize(self.crop_size)(sample)

        return sample

class RandomHorizontalFlip(object):
    """
    Randomly flip the input horizontally.
    Args:
        prob (float, optional): Probability of flipping the input. Defaults to .5.
    """

    def __init__(self, prob=0.5):
        super(RandomHorizontalFlip, self).__init__()
        self.prob = prob

    def apply_im(self, image):
        image = F.horizontal_flip(image)
        return image

    def apply_mask(self, mask):
        mask = F.horizontal_flip(mask)
        return mask

    def apply_bbox(self, bbox, width):
        oldx1 = bbox[:, 0].copy()
        oldx2 = bbox[:, 2].copy()
        bbox[:, 0] = width - oldx2
        bbox[:, 2] = width - oldx1
        return bbox

    def apply_segm(self, segms, height, width):
        flipped_segms = []
        for segm in segms:
            if F.is_poly(segm):
                # Polygon format
                flipped_segms.append(
                    [F.horizontal_flip_poly(poly, width) for poly in segm])
            else:
                # RLE format
                flipped_segms.append(F.horizontal_flip_rle(segm, height, width))
        return flipped_segms

    def apply(self, sample):
        if random.random() < self.prob:
            im_h, im_w = sample['image'].shape[:2]
            sample['image'] = self.apply_im(sample['image'])
            if 'image2' in sample:
                sample['image2'] = self.apply_im(sample['image2'])
            if 'mask' in sample:
                sample['mask'] = self.apply_mask(sample['mask'])
            if 'aux_masks' in sample:
                sample['aux_masks'] = list(
                    map(self.apply_mask, sample['aux_masks']))
            if 'gt_bbox' in sample and len(sample['gt_bbox']) > 0:
                sample['gt_bbox'] = self.apply_bbox(sample['gt_bbox'], im_w)
            if 'gt_poly' in sample and len(sample['gt_poly']) > 0:
                sample['gt_poly'] = self.apply_segm(sample['gt_poly'], im_h,
                                                    im_w)
            if 'target' in sample:
                sample['target'] = self.apply_im(sample['target'])
        return sample


class RandomVerticalFlip(object):
    """
    Randomly flip the input vertically.
    Args:
        prob (float, optional): Probability of flipping the input. Defaults to .5.
    """

    def __init__(self, prob=0.5):
        super(RandomVerticalFlip, self).__init__()
        self.prob = prob

    def apply_im(self, image):
        image = F.vertical_flip(image)
        return image

    def apply_mask(self, mask):
        mask = F.vertical_flip(mask)
        return mask

    def apply_bbox(self, bbox, height):
        oldy1 = bbox[:, 1].copy()
        oldy2 = bbox[:, 3].copy()
        bbox[:, 0] = height - oldy2
        bbox[:, 2] = height - oldy1
        return bbox

    def apply_segm(self, segms, height, width):
        flipped_segms = []
        for segm in segms:
            if F.is_poly(segm):
                # Polygon format
                flipped_segms.append(
                    [F.vertical_flip_poly(poly, height) for poly in segm])
            else:
                # RLE format
                flipped_segms.append(F.vertical_flip_rle(segm, height, width))
        return flipped_segms

    def apply(self, sample):
        if random.random() < self.prob:
            im_h, im_w = sample['image'].shape[:2]
            sample['image'] = self.apply_im(sample['image'])
            if 'image2' in sample:
                sample['image2'] = self.apply_im(sample['image2'])
            if 'mask' in sample:
                sample['mask'] = self.apply_mask(sample['mask'])
            if 'aux_masks' in sample:
                sample['aux_masks'] = list(
                    map(self.apply_mask, sample['aux_masks']))
            if 'gt_bbox' in sample and len(sample['gt_bbox']) > 0:
                sample['gt_bbox'] = self.apply_bbox(sample['gt_bbox'], im_h)
            if 'gt_poly' in sample and len(sample['gt_poly']) > 0:
                sample['gt_poly'] = self.apply_segm(sample['gt_poly'], im_h,
                                                    im_w)
            if 'target' in sample:
                sample['target'] = self.apply_im(sample['target'])
        return sample