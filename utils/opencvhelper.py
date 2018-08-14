#!/usr/bin/env python
"""
Copyright 2018, Zixin Luo, HKUST.
OpenCV helper.
"""

from __future__ import print_function

from threading import Thread
from Queue import Queue

import numpy as np

import cv2


class SiftWrapper(object):
    """"OpenCV SIFT wrapper."""

    def __init__(self, nfeatures=0, n_octave_layers=3,
                 peak_thld=0.0067, edge_thld=10, sigma=1.6,
                 n_sample=8192, patch_size=32):
        self.sift = None

        self.nfeatures = nfeatures
        self.n_octave_layers = n_octave_layers
        self.peak_thld = peak_thld
        self.edge_thld = edge_thld
        self.sigma = sigma
        self.n_sample = n_sample
        self.down_octave = True

        self.sift_init_sigma = 0.5
        self.sift_descr_scl_fctr = 3.
        self.sift_descr_width = 4

        self.first_octave = None
        self.max_octave = None
        self.pyr = None

        self.patch_size = patch_size
        self.output_gird = None

    def create(self):
        """Create OpenCV SIFT detector."""
        self.sift = cv2.xfeatures2d.SIFT_create(
            self.nfeatures, self.n_octave_layers, self.peak_thld, self.edge_thld, self.sigma)

    def detect(self, gray_img):
        """Detect keypoints in the gray-scale image.
        Args:
            gray_img: The input gray-scale image.
        Returns:
            npy_kpts: (n_kpts, 6) Keypoints represented as NumPy array.
            cv_kpts: A list of keypoints represented as cv2.KeyPoint.
        """

        cv_kpts = self.sift.detect(gray_img, None)

        all_octaves = [np.int8(i.octave & 0xFF) for i in cv_kpts]
        self.first_octave = int(np.min(all_octaves))
        self.max_octave = int(np.max(all_octaves))

        npy_kpts, cv_kpts = sample_by_octave(cv_kpts, self.n_sample, self.down_octave)
        return npy_kpts, cv_kpts

    def compute(self, img, cv_kpts):
        """Compute SIFT descriptions on given keypoints.
        Args:
            img: The input image, can be either color or gray-scale.
            cv_kpts: A list of cv2.KeyPoint.
        Returns:
            sift_desc: (n_kpts, 128) SIFT descriptions.
        """

        _, sift_desc = self.sift.compute(img, cv_kpts)
        return sift_desc

    def build_pyramid(self, gray_img):
        """Build pyramid. It would be more efficient to use the pyramid 
        constructed in the detection step.
        Args:
            gray_img: Input gray-scale image.
        Returns:
            pyr: A list of gaussian blurred images (gaussian scale space).
        """

        gray_img = gray_img.astype(np.float32)
        n_octaves = self.max_octave - self.first_octave + 1
        # create initial image.
        if self.first_octave < 0:
            sig_diff = np.sqrt(np.maximum(
                np.square(self.sigma) - np.square(self.sift_init_sigma) * 4, 0.01))
            base = cv2.resize(gray_img, (gray_img.shape[1] * 2, gray_img.shape[0] * 2),
                              interpolation=cv2.INTER_LINEAR)
            base = cv2.GaussianBlur(base, None, sig_diff)
        else:
            sig_diff = np.sqrt(np.maximum(np.square(self.sigma) -
                                          np.square(self.sift_init_sigma), 0.01))
            base = cv2.GaussianBlur(gray_img, None, sig_diff)
        # compute gaussian kernels.
        sig = np.zeros((self.n_octave_layers + 3,))
        self.pyr = [None] * (n_octaves * (self.n_octave_layers + 3))
        sig[0] = self.sigma
        k = np.power(2, 1. / self.n_octave_layers)
        for i in range(1, self.n_octave_layers + 3):
            sig_prev = np.power(k, i - 1) * self.sigma
            sig_total = sig_prev * k
            sig[i] = np.sqrt(sig_total * sig_total - sig_prev * sig_prev)
        # construct gaussian scale space.
        for o in range(0, n_octaves):
            for i in range(0, self.n_octave_layers + 3):
                if o == 0 and i == 0:
                    dst = base
                elif i == 0:
                    src = self.pyr[(o - 1) * (self.n_octave_layers + 3) + self.n_octave_layers]
                    dst = cv2.resize(
                        src, (src.shape[1] / 2, src.shape[0] / 2), interpolation=cv2.INTER_NEAREST)
                else:
                    src = self.pyr[o * (self.n_octave_layers + 3) + i - 1]
                    dst = cv2.GaussianBlur(src, None, sig[i])
                self.pyr[o * (self.n_octave_layers + 3) + i] = dst

    def unpack_octave(self, kpt):
        """Get scale coefficients of a keypoints.
        Args:
            kpt: A keypoint object represented as cv2.KeyPoint.
        Returns:
            octave: The octave index.
            layer: The level index.
            scale: The sampling step.
        """

        octave = kpt.octave & 255
        layer = (kpt.octave >> 8) & 255
        octave = octave if octave < 128 else (-128 | octave)
        scale = 1. / (1 << octave) if octave >= 0 else float(1 << -octave)
        return octave, layer, scale

    def get_interest_region(self, kpt_queue, all_patches, standardize=True):
        """Get the interest region around a keypoint.
        Args:
            kpt_queue: A queue to produce keypoint.
            all_patches: A list of cropped patches.
            standardize: (True by default) Whether to standardize patches as network inputs.
        Returns:
            Nothing.
        """
        while True:
            idx, cv_kpt = kpt_queue.get()
            # preprocess
            octave, layer, scale = self.unpack_octave(cv_kpt)
            size = cv_kpt.size * scale * 0.5
            ptf = (cv_kpt.pt[0] * scale, cv_kpt.pt[1] * scale)
            scale_img = self.pyr[(int(octave) - self.first_octave) *
                                 (self.n_octave_layers + 3) + int(layer)]
            ori = (360. - cv_kpt.angle) * (np.pi / 180.)
            radius = np.round(self.sift_descr_scl_fctr * size * np.sqrt(2)
                              * (self.sift_descr_width + 1) * 0.5)
            radius = np.minimum(radius, np.sqrt(np.sum(np.square(scale_img.shape))))
            # construct affine transformation matrix.
            affine_mat = np.zeros((3, 2), dtype=np.float32)
            m_cos = np.cos(ori) * radius
            m_sin = np.sin(ori) * radius
            affine_mat[0, 0] = m_cos
            affine_mat[1, 0] = m_sin
            affine_mat[2, 0] = ptf[0]
            affine_mat[0, 1] = -m_sin
            affine_mat[1, 1] = m_cos
            affine_mat[2, 1] = ptf[1]
            # get input grid.
            input_grid = np.matmul(self.output_grid, affine_mat)
            # sample image pixels.
            patch = cv2.remap(scale_img.astype(np.float32), np.reshape(input_grid, (-1, 1, 2)),
                              None, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
            patch = np.reshape(patch, (self.patch_size, self.patch_size))
            # standardize patches.
            if standardize:
                patch = (patch - np.mean(patch)) / (np.std(patch) + 1e-8)
            all_patches[idx] = patch
            kpt_queue.task_done()

    def get_patches(self, cv_kpts):
        """Get all patches around given keypoints.
        Args:
            cv_kpts: A list of keypoints represented as cv2.KeyPoint.
        Return:
            all_patches: (n_kpts, 32, 32) Cropped patches.
        """

        # generate sampling grids.
        n_pixel = np.square(self.patch_size)
        self.output_grid = np.zeros((n_pixel, 3), dtype=np.float32)
        for i in range(n_pixel):
            self.output_grid[i, 0] = (i % self.patch_size) * 1. / self.patch_size * 2 - 1
            self.output_grid[i, 1] = (i / self.patch_size) * 1. / self.patch_size * 2 - 1
            self.output_grid[i, 2] = 1

        all_patches = [None] * len(cv_kpts)
        # parallel patch cropping.
        kpt_queue = Queue()
        for i in range(4):
            worker_thread = Thread(target=self.get_interest_region, args=(kpt_queue, all_patches))
            worker_thread.daemon = True
            worker_thread.start()

        for idx, val in enumerate(cv_kpts):
            kpt_queue.put((idx, val))

        kpt_queue.join()
        all_patches = np.stack(all_patches)
        return all_patches


class MatcherWrapper(object):
    """OpenCV matcher wrapper."""

    def __init__(self):
        self.matcher = cv2.BFMatcher(cv2.NORM_L2)

    def get_matches(self, feat1, feat2, cv_kpts1, cv_kpts2, ratio=None, cross_check=True, info=''):
        """Compute putative and inlier matches.
        Args:
            feat: (n_kpts, 128) Local features.
            cv_kpts: A list of keypoints represented as cv2.KeyPoint.
            ratio: The threshold to apply ratio test.
            cross_check: (True by default) Whether to apply cross check.
            info: Info to print out.
        Returns:
            good_matches: Putative matches.
            mask: The mask to distinguish inliers/outliers on putative matches.
        """

        init_matches1 = self.matcher.knnMatch(feat1, feat2, k=2)
        init_matches2 = self.matcher.knnMatch(feat2, feat1, k=2)

        good_matches = []

        for i in range(len(init_matches1)):
            # cross check
            if cross_check and init_matches2[init_matches1[i][0].trainIdx][0].trainIdx == i:
                # ratio test
                if ratio is not None and init_matches1[i][0].distance <= ratio * init_matches1[i][1].distance:
                    good_matches.append(init_matches1[i][0])
                elif ratio is None:
                    good_matches.append(init_matches1[i][0])
            elif not cross_check:
                good_matches.append(init_matches1[i][0])

        good_kpts1 = np.array([cv_kpts1[m.queryIdx].pt for m in good_matches])
        good_kpts2 = np.array([cv_kpts2[m.trainIdx].pt for m in good_matches])

        _, mask = cv2.findFundamentalMat(good_kpts1, good_kpts2, cv2.RANSAC, 4.0, confidence=0.999)
        n_inlier = np.count_nonzero(mask)
        print(info, 'n_putative', len(good_matches), 'n_inlier', n_inlier)
        return good_matches, mask

    def draw_matches(self, img1, cv_kpts1, img2, cv_kpts2, good_matches, mask,
                     match_color=(0, 255, 0), pt_color=(0, 0, 255)):
        """Draw matches."""
        display = cv2.drawMatches(img1, cv_kpts1, img2, cv_kpts2, good_matches,
                                  None,
                                  matchColor=match_color,
                                  singlePointColor=pt_color,
                                  matchesMask=mask.ravel().tolist(), flags=4)
        return display


def sample_by_octave(cv_kpts, n_sample, down_octave=True):
    """Sample keypoints by octave.
    Args:
        cv_kpts: The list of keypoints representd as cv2.KeyPoint.
        n_sample: The sampling number of keypoint. Leave to -1 if no sampling needed
        down_octave: (True by default) Perform sampling downside of octave.
    Returns:
        npy_kpts: (n_kpts, 5) Keypoints in NumPy format, represenetd as
                  (x, y, size, orientation, octave).
        cv_kpts: A list of sampled cv2.KeyPoint.
    """

    n_kpts = len(cv_kpts)
    npy_kpts = np.zeros((n_kpts, 5))
    for idx, val in enumerate(cv_kpts):
        npy_kpts[idx, 0] = val.pt[0]
        npy_kpts[idx, 1] = val.pt[1]
        npy_kpts[idx, 2] = val.size
        npy_kpts[idx, 3] = val.angle * np.pi / 180.
        npy_kpts[idx, 4] = np.int8(val.octave & 0xFF)

    if down_octave:
        sort_idx = (-npy_kpts[:, 2]).argsort()
    else:
        sort_idx = (npy_kpts[:, 2]).argsort()

    npy_kpts = npy_kpts[sort_idx]
    cv_kpts = [cv_kpts[i] for i in sort_idx]

    if n_sample > -1 and n_kpts > n_sample:
        # get the keypoint number in each octave.
        _, unique_counts = np.unique(npy_kpts[:, 4], return_counts=True)

        if down_octave:
            unique_counts = list(reversed(unique_counts))

        n_keep = 0
        for i in unique_counts:
            if n_keep < n_sample:
                n_keep += i
            else:
                break
        print('Sampled', n_keep, 'from', n_kpts)
        npy_kpts = npy_kpts[:n_keep]
        cv_kpts = cv_kpts[:n_keep]

    return npy_kpts, cv_kpts
