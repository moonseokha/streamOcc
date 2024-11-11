import torch
import numpy as np
import cv2

from .occ_head_plugin import calculate_birds_eye_view_parameters
from ..utils import box3d_to_corners
from mmdet.datasets.builder import PIPELINES
import os

@PIPELINES.register_module()
class GenerateOccFlowLabels(object):
    def __init__(self, grid_conf, ignore_index=255, only_vehicle=True, filter_invisible=True, deal_instance_255=False,downsample=1):
        self.grid_conf = grid_conf
        # self.bev_resolution, self.bev_start_position, self.bev_dimension = calculate_birds_eye_view_parameters(
        #     grid_conf['x'], grid_conf['y'], grid_conf['z'],
        # )
        self.bev_resolution = np.array([grid_conf['x'][-1]*downsample,grid_conf['y'][-1]*downsample,grid_conf['z'][-1]])
        
        # convert numpy
        # self.bev_resolution = self.bev_resolution.numpy()
        # self.bev_start_position = self.bev_start_position.numpy()

        self.bev_dimension = np.array([
                                       int((grid_conf['x'][1]-grid_conf['x'][0])/self.bev_resolution[0]),
                                       int((grid_conf['y'][1]-grid_conf['y'][0])/self.bev_resolution[1]),
                                       int((grid_conf['z'][1]-grid_conf['z'][0])/self.bev_resolution[2])
                                       ])
        self.bev_start_position = np.array([
                                            grid_conf['x'][0], 
                                            grid_conf['y'][0],
                                            grid_conf['z'][0],
                                            ]
                                           )
        self.ignore_index = ignore_index
        self.only_vehicle = only_vehicle
        self.filter_invisible = filter_invisible
        self.deal_instance_255 = deal_instance_255
        self.downsample = downsample
        assert self.deal_instance_255 is False
        
        # nusc_classes = ['car', 'truck', 'construction_vehicle', 'bus', 'trailer',
        #                 'barrier', 'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone']
     


    def __call__(self, results):
        """
        # Given lidar frame bboxes for curr frame and each future frame,
        # generate segmentation, instance, centerness, offset, and fwd flow map
        """
        # Avoid ignoring obj with index = self.ignore_index
        SPECIAL_INDEX = -20
    
        
        instances = []
        gt_future_boxes = []
        gt_future_labels = []

        gt_bboxes_3d, gt_labels_3d = results['gt_bboxes_3d'],results['gt_labels_3d']
        segmentations = np.zeros(
                (self.bev_dimension[1],self.bev_dimension[0],self.bev_dimension[2]))
        if gt_bboxes_3d is not None:
            # valid sample and has objects
            if len(gt_bboxes_3d) > 0:                    
                bbox_corners =  box3d_to_corners(gt_bboxes_3d)[:, :, :3]
                bbox_bottom = bbox_corners[:, [
                    0, 3, 7, 4], :2]
                # print("self.bev_start_position[:2]: ",self.bev_start_position[:2])
                bbox_bottom = np.round(
                    (bbox_bottom - self.bev_start_position[:2]) / self.bev_resolution[:2]).astype(np.int32)
                bbox_z = np.round(
                    (bbox_corners[...,:2:3] - self.bev_start_position[2:3]) / self.bev_resolution[2:3]).astype(np.int32)
                

                
                for index, gt_label in enumerate(gt_labels_3d):
                    segmentation = np.zeros(
                        (self.bev_dimension[1], self.bev_dimension[0]))
                    
                    # if gt_ind == self.ignore_index:
                    #     gt_ind = SPECIAL_INDEX   # 255 -> -20
                    poly_region = bbox_bottom[index]
                    box_z = bbox_z[index]
                    min_z = np.min(box_z)
                    max_z = np.max(box_z)
                    cv2.fillPoly(segmentation, [np.array(poly_region)], 1.0)
                    # print("segmentation.sum(): ", segmentation.sum())
                    segmentations[..., min_z:max_z+1] = np.logical_or(segmentations[..., min_z:max_z+1], segmentation[..., np.newaxis])


                # for index, gt_label in enumerate(gt_labels_3d):
                #     segmentations = np.zeros(self.bev_dimension)
                #     segmentation = np.zeros(
                #         (self.bev_dimension[1], self.bev_dimension[0]))
                    
                #     # if gt_ind == self.ignore_index:
                #     #     gt_ind = SPECIAL_INDEX   # 255 -> -20
                #     poly_region = bbox_bottom[index]
                #     box_z = bbox_z[index]
                #     min_z = np.min(box_z)
                #     max_z = np.max(box_z)
                #     cv2.fillPoly(segmentation, [np.array(poly_region)], 1.0)
                #     # print("segmentation.sum(): ", segmentation.sum())
                #     segmentations[..., min_z:max_z+1] = np.logical_or(segmentations[..., min_z:max_z+1], segmentation[..., np.newaxis])
                #     occupied_indexes = np.where(segmentations==1)
                #     occupied_indexes = torch.stack([torch.from_numpy(occupied_indexes[0]),torch.from_numpy(occupied_indexes[1]),torch.from_numpy(occupied_indexes[2])],dim=-1)
                #     occupied_flatten_indexes = occupied_indexes[0]*occupied_indexes[1]*self.bev_dimension[2] + occupied_indexes[1]*self.bev_dimension[2] + occupied_indexes[2]



        # segmentation = 1 where objects are located
        segmentations = torch.from_numpy(
            segmentations).long()
  
        results.update({
            'gt_segmentation': segmentations,
        })
        return results
