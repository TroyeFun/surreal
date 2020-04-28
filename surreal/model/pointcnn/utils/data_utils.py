from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import h5py
import plyfile
import numpy as np
from matplotlib import cm
import scipy.spatial.distance as distance
import cv2
import torch


def save_ply(points, filename, colors=None, normals=None):
    vertex = np.array([tuple(p) for p in points], dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4')])
    n = len(vertex)
    desc = vertex.dtype.descr

    if normals is not None:
        vertex_normal = np.array([tuple(n) for n in normals], dtype=[('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4')])
        assert len(vertex_normal) == n
        desc = desc + vertex_normal.dtype.descr

    if colors is not None:
        vertex_color = np.array([tuple(c * 255) for c in colors],
                                dtype=[('red', 'u1'), ('green', 'u1'), ('blue', 'u1')])
        assert len(vertex_color) == n
        desc = desc + vertex_color.dtype.descr

    vertex_all = np.empty(n, dtype=desc)

    for prop in vertex.dtype.names:
        vertex_all[prop] = vertex[prop]

    if normals is not None:
        for prop in vertex_normal.dtype.names:
            vertex_all[prop] = vertex_normal[prop]

    if colors is not None:
        for prop in vertex_color.dtype.names:
            vertex_all[prop] = vertex_color[prop]

    ply = plyfile.PlyData([plyfile.PlyElement.describe(vertex_all, 'vertex')], text=False)
    if not os.path.exists(os.path.dirname(filename)):
        os.makedirs(os.path.dirname(filename))
    ply.write(filename)


def save_ply_property(points, property, property_max, filename, cmap_name='Set1'):
    point_num = points.shape[0]
    colors = np.full(points.shape, 0.5)
    cmap = cm.get_cmap(cmap_name)
    for point_idx in range(point_num):
        colors[point_idx] = cmap(property[point_idx] / property_max)[:3]
    save_ply(points, filename, colors)


def save_ply_batch(points_batch, file_path, points_num=None):
    batch_size = points_batch.shape[0]
    if type(file_path) != list:
        basename = os.path.splitext(file_path)[0]
        ext = '.ply'
    for batch_idx in range(batch_size):
        point_num = points_batch.shape[1] if points_num is None else points_num[batch_idx]
        if type(file_path) == list:
            save_ply(points_batch[batch_idx][:point_num], file_path[batch_idx])
        else:
            save_ply(points_batch[batch_idx][:point_num], '%s_%04d%s' % (basename, batch_idx, ext))


def save_ply_property_batch(points_batch, property_batch, file_path, points_num=None, property_max=None,
                            cmap_name='Set1'):
    batch_size = points_batch.shape[0]
    if type(file_path) != list:
        basename = os.path.splitext(file_path)[0]
        ext = '.ply'
    property_max = np.max(property_batch) if property_max is None else property_max
    for batch_idx in range(batch_size):
        point_num = points_batch.shape[1] if points_num is None else points_num[batch_idx]
        if type(file_path) == list:
            save_ply_property(points_batch[batch_idx][:point_num], property_batch[batch_idx][:point_num],
                              property_max, file_path[batch_idx], cmap_name)
        else:
            save_ply_property(points_batch[batch_idx][:point_num], property_batch[batch_idx][:point_num],
                              property_max, '%s_%04d%s' % (basename, batch_idx, ext), cmap_name)


def save_ply_point_with_normal(data_sample, folder):
    for idx, sample in enumerate(data_sample):
        filename_pts = os.path.join(folder, '{:08d}.ply'.format(idx))
        save_ply(sample[..., :3], filename_pts, normals=sample[..., 3:])


def grouped_shuffle(inputs):
    for idx in range(len(inputs) - 1):
        assert (len(inputs[idx]) == len(inputs[idx + 1]))

    shuffle_indices = np.arange(inputs[0].shape[0])
    np.random.shuffle(shuffle_indices)
    outputs = []
    for idx in range(len(inputs)):
        outputs.append(inputs[idx][shuffle_indices, ...])
    return outputs


def load_cls(filelist):
    points = []
    labels = []

    folder = os.path.dirname(filelist)
    for line in open(filelist):
        filename = os.path.basename(line.rstrip())
        data = h5py.File(os.path.join(folder, filename))
        if 'normal' in data:
            points.append(np.concatenate([data['data'][...], data['data'][...]], axis=-1).astype(np.float32))
        else:
            points.append(data['data'][...].astype(np.float32))
        labels.append(np.squeeze(data['label'][:]).astype(np.int32))
    return (np.concatenate(points, axis=0),
            np.concatenate(labels, axis=0))


def load_cls_train_val(filelist, filelist_val):
    data_train, label_train = grouped_shuffle(load_cls(filelist))
    data_val, label_val = load_cls(filelist_val)
    return data_train, label_train, data_val, label_val


def load_seg(filelist):
    points = []
    labels = []
    point_nums = []
    labels_seg = []

    folder = os.path.dirname(filelist)
    for line in open(filelist):
        filename = os.path.basename(line.rstrip())
        data = h5py.File(os.path.join(folder, filename))
        points.append(data['data'][...].astype(np.float32))
        labels.append(data['label'][...].astype(np.int32))
        point_nums.append(data['data_num'][...].astype(np.int32))
        labels_seg.append(data['label_seg'][...].astype(np.int32))
    return (np.concatenate(points, axis=0),
            np.concatenate(labels, axis=0),
            np.concatenate(point_nums, axis=0),
            np.concatenate(labels_seg, axis=0))


def sample_pcd(pcd, num_points):
    # upsample
    if pcd.shape[0] < num_points:
        n_repeat = num_points // pcd.shape[0]
        min_pt, max_pt = pcd.min(dim=0)[0], pcd.max(dim=0)[0]
        scale = torch.norm(max_pt-min_pt) / 10  # what about cases that pcd contains only one point

        pcds = [pcd]
        for t in range(n_repeat):
            new_pcd = pcd + torch.randn_like(pcd) * scale
            pcds.append(new_pcd)
        pcd = torch.concatenate(pcds, dim=0)

    # downsample
    idxs = torch.randperm(pcd.shape[0])[:num_points]
    pcd = pcd[idxs]
    return pcd
    

def get_random_pcd(num_points, range=(-1,1)):
    pcd = torch.rand((num_points, 3)) * (range[1]-range[0]) + range[0]
    return pcd


class Pix2PCD:

    def __init__(self, camera_mat, camera_pos, camera_f, num_points=128):
        """
        camera_mat: 4x4 np.array
        camera_f: float
        color: ...
        have not supported cpu
        """
        self.camera_mat = torch.from_numpy(camera_mat).cuda().view(3,3)
        self.camera_pos = torch.from_numpy(camera_pos).cuda().view(3,1)
        self.camera_f = camera_f
        self.num_points = num_points
        self.hsv_range = {
            'blue': [[100, 150, 150], [124, 255, 255]],  # [lower, upper]
        }

    def __call__(self, rgbd_img_batch, color):
        """would this part be time-comsuming?"""
        lower, upper = np.array(self.hsv_range[color][0]), np.array(self.hsv_range[color][1])
        print('debug: rgbd_img_batch shape: ', rgbd_img_batch.shape)

        color_imgs, depth_imgs = rgbd_img_batch[:, :3, :, :], rgbd_img_batch[:, 3, :, :]
        color_imgs = color_imgs.permute(0, 2, 3, 1)
        color_imgs_np = color_imgs.cpu().numpy()

        n, w, h, _ = color_imgs.shape

        # calculate (x, y) in pixel coordinate
        x_pix = torch.arange(w).cuda() - (w - 1)/2
        x_pix = x_pix.view(1, -1).repeat(h, 1)
        y_pix = torch.arange(h).cuda() - (h - 1)/2
        y_pix = y_pix.view(-1, 1).repeat(1, w)

        pcds = []
        for idx in range(n):
            depth_img = depth_imgs[idx]
            hsv_img = cv2.cvtColor(color_imgs_np[idx], cv2.COLOR_BGR2HSV)
            mask = cv2.inRange(hsv_img, lower, upper)
            mask = torch.from_numpy(mask).cuda()
            mask_index = mask > 0
    
            print('debug: {} points detected'.format(mask_index.sum().item()))
            if mask_index.sum() > 0:
                x_pcd = x_pix * depth_img / self.camera_f
                y_pcd = y_pix * depth_img / self.camera_f

                # apply mask
                x_pcd = x_pcd[mask_index]
                y_pcd = -y_pcd[mask_index]
                z_pcd = -depth_img[mask_index]
                pcd = torch.stack([x_pcd, y_pcd, z_pcd], dim=0)
                pcd = self.camera_mat.mm(pcd) + self.camera_pos
                pcd = pcd.transpose(0, 1)
                pcd = sample_pcd(pcd, self.num_points)
            else:
                """no points detected"""
                pcd = get_random_pcd(self.num_points)

            pcds.append(pcd)
        pcds = torch.stack(pcds, dim=0)
        return pcds 
            
