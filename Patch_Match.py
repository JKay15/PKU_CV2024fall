# patchmatch algorithm
# given two images, find the best match for each patch in the source image

import numpy as np
import cv2
import random
import time


class PatchMatch(object):
    """
    PatchMatch class
    """
    def __init__(self, A, B, patch_size):
        # assert A.shape == A_.shape == B.shape == B_.shape, "Unequal dimensions for patch-matching input"
        self.A = self.tensor_to_numpy(A)
        self.B = self.tensor_to_numpy(B)
        self.patch_size = patch_size
        # nnf: nearest neighbour field, shape (H, W, 2)
        # nnf[i, j] = [x, y] where (i, j) is the pixel in A and (x, y) is the nearest neighbour in B
        self.nnf = np.zeros(shape=(self.A.shape[0], self.A.shape[1], 2)).astype(np.int64)
        # nnd: nearest neighbour distance, shape (H, W)
        self.nnd = np.zeros(shape=(self.A.shape[0], self.A.shape[1]))
        self.nnf_init()

    def tensor_to_numpy(self, tensor):
        # tensor: (1, C, H, W)
        # numpy: (H, W, C)
        tensor = tensor.squeeze(0).cpu()
        return tensor.numpy().transpose(1, 2, 0)

    def nnf_init(self):
        """
        Randomly initialize nnd and calculate nnd
        """
        hA, wA, cA = self.A.shape
        hB, wB, cB = self.B.shape
        self.nnf[:, :, 0] = np.random.randint(hB, size=(hA, wA))
        self.nnf[:, :, 1] = np.random.randint(wB, size=(hA, wA))
        for i in range(hA):
            for j in range(wA):
                x, y = self.nnf[i, j]
                self.nnd[i, j] = self.cal_dist(i, j, x, y)

    def cal_dist(self, i, j, x, y):
        """
        Calculate distance between patch A[i, j] and patch B[x, y]
              dxu
              |
        dyl--x, y--dyr
              |
              dxd
        """
        dxu = dyl = self.patch_size // 2
        dxd = dyr = self.patch_size // 2 + 1  # +1 to include the center pixel
        dxu = min(i, x, dxu)
        dxd = min(self.A.shape[0] - i, self.B.shape[0] - x, dxd)
        dyl = min(j, y, dyl)
        dyr = min(self.A.shape[1] - j, self.B.shape[1] - y, dyr)
        patch_A = self.A[i - dxu: i + dxd, j - dyl: j + dyr]
        patch_B = self.B[x - dxu: x + dxd, y - dyl: y + dyr]
        return np.sum((patch_A - patch_B) ** 2) / ((dxu + dxd) * (dyl + dyr))

    def reconstruct(self, img_A, img_B):
        """
        Assume we have got the nnf, reconstruct the image using the nnf correspondence
        """
        h, w, c = img_A.shape
        h_nn, w_nn, _ = self.nnf.shape
        factor = h // h_nn  # scale factor, since nnf is done on smaller feature maps
        output = np.zeros_like(img_A)
        for i in range(h_nn):
            for j in range(w_nn):
                x, y = self.nnf[i, j]
                if (output[factor * i: factor * (i + 1), factor * j: factor * (j + 1)].shape
                    == img_B[factor * x: factor * (x + 1), factor * y: factor * (y + 1)].shape):
                    output[factor * i: factor * (i + 1), factor * j: factor * (j + 1)] = \
                        img_B[factor * x: factor * (x + 1), factor * y: factor * (y + 1)]
        return output

    # def reconstruct_avg(self, img_A, img_B):
    #     """
    #     Reconstruct image using average voting.
    #           dxu
    #           |
    #     dyl--x, y--dyr
    #           |
    #           dxd
    #     """
    #     h, w, c = img_A.shape
    #     output = np.zeros_like(img_A)
    #     for i in range(h):
    #         for j in range(w):
    #             dxu = dyl = self.patch_size // 2
    #             dxd = dyr = self.patch_size // 2 + 1
    #             dxu = min(i, dxu)
    #             dxd = min(h - i, dxd)
    #             dyl = min(j, dyl)
    #             dyr = min(w - j, dyr)
    #             patch_nnf = self.nnf[i - dxu: i + dxd, j - dyl: j + dyr]
    #             h_p, w_p, _ = patch_nnf.shape
    #             lookups = np.zeros(shape=(h_p, w_p, c), dtype=np.float32)
    #
    #             # create lookups for averaging
    #             for m in range(h_p):
    #                 for n in range(w_p):
    #                     x, y = patch_nnf[m, n]
    #                     lookups[m, n] = img_B[x, y]
    #
    #             # average voting
    #             if lookups.size > 0:
    #                 output[i, j] = np.mean(lookups, axis=(0, 1))
    #     return output

    # def upsample(self, size):
    #     """
    #     Upsample the nnf to a larger size
    #     """
    #     h, w, p = self.nnf.shape
    #     nnf_trans = np.zeros(shape=(h, w, 3))  # nnf[i, j] = [x, y, 0], x, y = self.nnf[i, j]
    #     # this nnf is created to facilitate nearest neighbour interpolation
    #     for i in range(h):
    #         for j in range(w):
    #             nnf_trans[i, j] = [self.nnf[i, j][0], self.nnf[i, j][1], 0]
    #     factor = size // h  # scale factor
    #     nnf_up = np.zeros(shape=(size, size, 2)).astype(np.int64)
    #     # nearest neighbour interpolation
    #     nnf_trans_up = cv2.resize(nnf_trans, None, fx=factor, fy=factor, interpolation=cv2.INTER_NEAREST)
    #     for i in range(nnf_trans_up.shape[0]):
    #         for j in range(nnf_trans_up.shape[1]):
    #             x, y, _ = nnf_trans_up[i, j]  # new x, y
    #             nnf_up[i, j] = [x * factor, y * factor]
    #     return nnf_up

    def propagate(self, iters, search_radius=None):
        """
        Propagate the nnf field
        """
        h, w, c = self.A.shape
        for i in range(iters):
            if i % 2 == 0:
                x_s = y_s = 0
                x_e, y_e = h, w
                dx = dy = 1
            else:  # reverse the direction of propagation
                x_e = y_e = -1
                x_s, y_s = h - 1, w - 1
                dx = dy = -1
            x = x_s
            while x != x_e:
                y = y_s
                while y != y_e:
                    self.propagate_pixel(x, y, dx, dy, search_radius)
                    y += dy
                x += dx
            print(f"Propagation iteration {i + 1} done")

    def propagate_pixel(self, i, j, di, dj, search_radius):
        hA, wA, cA = self.A.shape
        hB, wB, cB = self.B.shape
        x_nn, y_nn = self.nnf[i, j]
        best_dist = self.nnd[i, j]
        if 0 <= i - di < hA:  # check if the upper pixel is within the image
            x_up, y_up = self.nnf[i - di, j]
            x_up += di
            if 0 <= x_up < hB and 0 <= y_up < wB:
                dist_up = self.cal_dist(i, j, x_up, y_up)
                if dist_up < best_dist:
                    best_dist = dist_up
                    x_nn, y_nn = x_up, y_up
        if 0 <= j - dj < wA:  # check if the left pixel is within the image
            x_left, y_left = self.nnf[i, j - dj]
            y_left += dj
            if 0 <= x_left < hB and 0 <= y_left < wB:
                dist_left = self.cal_dist(i, j, x_left, y_left)
                if dist_left < best_dist:
                    best_dist = dist_left
                    x_nn, y_nn = x_left, y_left
        # random search
        r = search_radius if search_radius else max(hB, wB)
        while r >= 1:
            lower,upper=max(0, i - r), min(hB - 1, i + r)
            if lower>upper:lower,upper=upper,lower
            x_rand = random.randint(lower,upper)
            lower,upper=max(0, j - r), min(wB - 1, j + r)
            if lower>upper:lower,upper=upper,lower
            y_rand = random.randint(lower,upper)
            dist_rand = self.cal_dist(i, j, x_rand, y_rand)
            if dist_rand < best_dist:
                best_dist = dist_rand
                x_nn, y_nn = x_rand, y_rand
            r //= 2

        self.nnf[i, j] = [x_nn, y_nn]
        self.nnd[i, j] = best_dist

    def visualize(self):
        nnf = self.nnf

        img = np.zeros((nnf.shape[0], nnf.shape[1], 3), dtype=np.uint8)

        for i in range(nnf.shape[0]):
            for j in range(nnf.shape[1]):
                pos = nnf[i, j]
                img[i, j, 0] = int(255 * (pos[0] / self.B.shape[0]))
                img[i, j, 2] = int(255 * (pos[1] / self.B.shape[1]))

        return img