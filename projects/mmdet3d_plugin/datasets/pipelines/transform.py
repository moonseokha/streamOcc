import numpy as np
import mmcv
from mmcv.parallel import DataContainer as DC
from mmdet.datasets.builder import PIPELINES
from mmdet.datasets.pipelines import to_tensor
import pdb

@PIPELINES.register_module()
class MultiScaleDepthMapGenerator(object):
    def __init__(self, downsample=1, max_depth=60):
        if not isinstance(downsample, (list, tuple)):
            downsample = [downsample]
        self.downsample = downsample
        self.max_depth = max_depth

    def __call__(self, input_dict):
        points = input_dict["points"][..., :3, None]
        gt_depth = []
        for i, lidar2img in enumerate(input_dict["lidar2img"]):
            H, W = input_dict["img_shape"][i][:2]

            pts_2d = (
                np.squeeze(lidar2img[:3, :3] @ points, axis=-1)
                + lidar2img[:3, 3]
            )
            pts_2d[:, :2] /= pts_2d[:, 2:3]
            U = np.round(pts_2d[:, 0]).astype(np.int32)
            V = np.round(pts_2d[:, 1]).astype(np.int32)
            depths = pts_2d[:, 2]
            mask = np.logical_and.reduce(
                [
                    V >= 0,
                    V < H,
                    U >= 0,
                    U < W,
                    depths >= 0.1,
                    # depths <= self.max_depth,
                ]
            )
            V, U, depths = V[mask], U[mask], depths[mask]
            sort_idx = np.argsort(depths)[::-1]
            V, U, depths = V[sort_idx], U[sort_idx], depths[sort_idx]
            depths = np.clip(depths, 0.1, self.max_depth)
            for j, downsample in enumerate(self.downsample):
                if len(gt_depth) < j + 1:
                    gt_depth.append([])
                h, w = (int(H / downsample), int(W / downsample))
                u = np.floor(U / downsample).astype(np.int32)
                v = np.floor(V / downsample).astype(np.int32)
                depth_map = np.ones([h, w], dtype=np.float32) * -1
                depth_map[v, u] = depths
                gt_depth[j].append(depth_map)

        input_dict["gt_depth_det"] = [np.stack(x) for x in gt_depth]
        return input_dict


@PIPELINES.register_module()
class NuScenesSparse4DAdaptor(object):
    def __init(self):
        pass

    def __call__(self, input_dict):
        input_dict["projection_mat"] = np.float32(
            np.stack(input_dict["ego2sensors"])
        )
        input_dict["image_wh"] = np.ascontiguousarray(
            np.array(input_dict["img_shape"], dtype=np.float32)[:, :2][:, ::-1]
        )
        input_dict["T_global_inv"] = np.linalg.inv(input_dict["box2global"])
        input_dict["T_global"] = input_dict["box2global"]
        if "cam_intrinsic" in input_dict:
            input_dict["cam_intrinsic"] = np.float32(
                np.stack(input_dict["cam_intrinsic"])
            )
            input_dict["focal"] = input_dict["cam_intrinsic"][..., 0, 0]
            # input_dict["focal"] = np.sqrt(
            #     np.abs(np.linalg.det(input_dict["cam_intrinsic"][:, :2, :2]))
            # )
        if "instance_inds" in input_dict:
            input_dict["instance_id"] = input_dict["instance_inds"]
        if "gt_bboxes_3d" in input_dict:
            if len(input_dict["gt_bboxes_3d"])>0:
                input_dict["gt_bboxes_3d"][:, 6] = self.limit_period(
                    input_dict["gt_bboxes_3d"][:, 6], offset=0.5, period=2 * np.pi
                )
                input_dict["gt_bboxes_3d"] = DC(
                    to_tensor(input_dict["gt_bboxes_3d"]).float()
                )
        # if "future_gt_bboxes_3d" in input_dict:
        #     input_dict["future_gt_bboxes_3d"][:, -1] = self.limit_period(
        #         input_dict["future_gt_bboxes_3d"][:, -1], offset=0.5, period=2 * np.pi
        #     )
        #     input_dict["future_gt_bboxes_3d"] = DC(
        #         to_tensor(input_dict["future_gt_bboxes_3d"]).float()
        #     )
        if "gt_labels_3d" in input_dict:
            if len(input_dict["gt_labels_3d"])>0:
                input_dict["gt_labels_3d"] = DC(
                    to_tensor(input_dict["gt_labels_3d"]).long()
                )
        if "bda" in input_dict:
            input_dict["view_tran_comp"] = [input_dict["cam2coor"], np.float32(np.stack(input_dict["intrinsic_mat"])), input_dict['post_rots'],np.float32(np.stack(input_dict['post_trans'])),np.float32(np.stack(input_dict['bda']))]
        if "gt_segmentation" in input_dict:
            input_dict["gt_segmentation"] = DC(
                to_tensor(input_dict["gt_segmentation"]).long()
            )
        if "gt_occ" in input_dict:
            input_dict["gt_occ"] = DC(to_tensor(input_dict["gt_occ"]), stack=False)
            
        imgs = [img.transpose(2, 0, 1) for img in input_dict["img"]]
        imgs = np.ascontiguousarray(np.stack(imgs, axis=0))
        input_dict["img"] = DC(to_tensor(imgs), stack=True)
        return input_dict

    def limit_period(
        self, val: np.ndarray, offset: float = 0.5, period: float = np.pi
    ) -> np.ndarray:
        limited_val = val - np.floor(val / period + offset) * period
        return limited_val


@PIPELINES.register_module()
class InstanceNameFilter(object):
    """Filter GT objects by their names.

    Args:
        classes (list[str]): List of class names to be kept for training.
    """

    def __init__(self, classes):
        self.classes = classes
        self.labels = list(range(len(self.classes)))

    def __call__(self, input_dict):
        """Call function to filter objects by their names.

        Args:
            input_dict (dict): Result dict from loading pipeline.

        Returns:
            dict: Results after filtering, 'gt_bboxes_3d', 'gt_labels_3d' \
                keys are updated in the result dict.
        """
        gt_labels_3d = input_dict["gt_labels_3d"]
        gt_bboxes_mask = np.array(
            [n in self.labels for n in gt_labels_3d], dtype=np.bool_
        )
        # print("gt_bboxes_3d: ",input_dict["gt_bboxes_3d"].shape)
        # print("gt_labels_3d: ",input_dict["gt_labels_3d"].shape)
        # print("gt_bboxes_mask: ",gt_bboxes_mask.shape)
        # print("=====================================")
        input_dict["gt_bboxes_3d"] = input_dict["gt_bboxes_3d"][gt_bboxes_mask]
        input_dict["gt_labels_3d"] = input_dict["gt_labels_3d"][gt_bboxes_mask]
        if "instance_inds" in input_dict:
            input_dict["instance_inds"] = input_dict["instance_inds"][gt_bboxes_mask]

        return input_dict

    def __repr__(self):
        """str: Return a string that describes the module."""
        repr_str = self.__class__.__name__
        repr_str += f"(classes={self.classes})"
        return repr_str


@PIPELINES.register_module()
class CircleObjectRangeFilter(object):
    def __init__(
        self, class_dist_thred=[52.5] * 5 + [31.5] + [42] * 3 + [31.5], max_z = None, min_z = None,
    ):
        self.class_dist_thred = class_dist_thred
        self.max_z = max_z
        self.min_z = min_z

    def __call__(self, input_dict):
        if len(input_dict["gt_bboxes_3d"])==0:
            return input_dict
        gt_bboxes_3d = input_dict["gt_bboxes_3d"]
        gt_labels_3d = input_dict["gt_labels_3d"]
        # if 'future_gt_bboxes_3d' in input_dict:
        #     gt_future_bboxes_3d = input_dict["future_gt_bboxes_3d"]
        dist = np.sqrt(
            np.sum(gt_bboxes_3d[:, :2] ** 2, axis=-1)
        )
        mask = np.array([False] * len(dist))
        for label_idx, dist_thred in enumerate(self.class_dist_thred):
            mask = np.logical_or(
                mask,
                np.logical_and(gt_labels_3d == label_idx, dist <= dist_thred),
            )

        gt_bboxes_3d = gt_bboxes_3d[mask]
        gt_labels_3d = gt_labels_3d[mask]
        mask_z = None
        if self.max_z is not None and self.min_z is not None:
            top_z = gt_bboxes_3d[:, 2] + gt_bboxes_3d[:, 5] / 2
            bottom_z = gt_bboxes_3d[:, 2] - gt_bboxes_3d[:, 5] / 2
            mask_z = np.logical_and(top_z < self.max_z, bottom_z > self.min_z)
            gt_bboxes_3d = gt_bboxes_3d[mask_z]
            gt_labels_3d = gt_labels_3d[mask_z]
        input_dict["gt_bboxes_3d"] = gt_bboxes_3d
        input_dict["gt_labels_3d"] = gt_labels_3d
        # if 'future_gt_bboxes_3d' in input_dict:
        #     input_dict["future_gt_bboxes_3d"] = gt_future_bboxes_3d[mask]
        if "instance_inds" in input_dict:
            input_dict["instance_inds"] = input_dict["instance_inds"][mask]
            if mask_z is not None:
                input_dict["instance_inds"] = input_dict["instance_inds"][mask_z]

        return input_dict

    def __repr__(self):
        """str: Return a string that describes the module."""
        repr_str = self.__class__.__name__
        repr_str += f"(class_dist_thred={self.class_dist_thred})"
        return repr_str


@PIPELINES.register_module()
class NormalizeMultiviewImage(object):
    """Normalize the image.
    Added key is "img_norm_cfg".
    Args:
        mean (sequence): Mean values of 3 channels.
        std (sequence): Std values of 3 channels.
        to_rgb (bool): Whether to convert the image from BGR to RGB,
            default is true.
    """

    def __init__(self, mean, std, to_rgb=True):
        self.mean = np.array(mean, dtype=np.float32)
        self.std = np.array(std, dtype=np.float32)
        self.to_rgb = to_rgb

    def __call__(self, results):
        """Call function to normalize images.
        Args:
            results (dict): Result dict from loading pipeline.
        Returns:
            dict: Normalized results, 'img_norm_cfg' key is added into
                result dict.
        """
        results["img"] = [
            mmcv.imnormalize(img, self.mean, self.std, self.to_rgb)
            for img in results["img"]
        ]
        results["img_norm_cfg"] = dict(
            mean=self.mean, std=self.std, to_rgb=self.to_rgb
        )
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f"(mean={self.mean}, std={self.std}, to_rgb={self.to_rgb})"
        return repr_str
