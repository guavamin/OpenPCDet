import argparse
import glob
from pathlib import Path

# try:
#     import open3d
#     from visual_utils import open3d_vis_utils as V
#     OPEN3D_FLAG = True
# except:
#     import mayavi.mlab as mlab
#     from visual_utils import visualize_utils as V
#     OPEN3D_FLAG = False

import numpy as np
import torch
import time
from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.datasets import DatasetTemplate
from pcdet.models import build_network, load_data_to_gpu
from pcdet.utils import common_utils
import cv2
from torchvision.ops.boxes import nms as torch_nms
from motpy import Detection, MultiObjectTracker

# Rotating the point clouds to be consistent with the video
def rotation_matrix(theta1, theta2, theta3, order='xyz'):
    c1 = np.cos(theta1 * np.pi / 180)
    s1 = np.sin(theta1 * np.pi / 180)
    c2 = np.cos(theta2 * np.pi / 180)
    s2 = np.sin(theta2 * np.pi / 180)
    c3 = np.cos(theta3 * np.pi / 180)
    s3 = np.sin(theta3 * np.pi / 180)
    matrix = np.array([[c2 * c3, -c2 * s3, s2],
                       [c1 * s3 + c3 * s1 * s2, c1 * c3 - s1 * s2 * s3, -c2 * s1],
                       [s1 * s3 - c1 * c3 * s2, c3 * s1 + c1 * s2 * s3, c1 * c2]])
    return matrix

class DemoDataset(DatasetTemplate):
    def __init__(self, dataset_cfg, class_names, training=True, root_path=None, logger=None, ext='.bin'):
        """
        Args:
            root_path:
            dataset_cfg:
            class_names:
            training:
            logger:
        """
        super().__init__(
            dataset_cfg=dataset_cfg, class_names=class_names, training=training, root_path=root_path, logger=logger
        )
        self.root_path = root_path
        self.ext = ext
        data_file_list = glob.glob(str(root_path / f'*{self.ext}')) if self.root_path.is_dir() else [self.root_path]

        data_file_list.sort()
        self.sample_file_list = data_file_list

    def __len__(self):
        return len(self.sample_file_list)

    def __getitem__(self, index):
        if self.ext == '.bin':
            # When loading the .bin file in nuScenesc format, you need to manually change the reshape to (-1, 5)
            # Likely, for KITTI dataset, change it to (-1, 4)
            points = np.fromfile(self.sample_file_list[index], dtype=np.float32).reshape(-1, 5)
        elif self.ext == '.npy':
            points = np.load(self.sample_file_list[index])
        else:
            raise NotImplementedError

        input_dict = {
            'points': points,
            'frame_id': index,
        }

        data_dict = self.prepare_data(data_dict=input_dict)
        return data_dict


def parse_config():
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--cfg_file', type=str, default='cfgs/kitti_models/second.yaml',
                        help='specify the config for demo')
    parser.add_argument('--data_path', type=str, default='demo_data',
                        help='specify the point cloud data file or directory')
    parser.add_argument('--ckpt', type=str, default=None, help='specify the pretrained model')
    parser.add_argument('--ext', type=str, default='.bin', help='specify the extension of your point cloud data file')

    args = parser.parse_args()

    cfg_from_yaml_file(args.cfg_file, cfg)

    return args, cfg

def main():
    args, cfg = parse_config()
    logger = common_utils.create_logger()
    logger.info('-----------------Quick Demo of OpenPCDet-------------------------')
    demo_dataset = DemoDataset(
        dataset_cfg=cfg.DATA_CONFIG, class_names=cfg.CLASS_NAMES, training=False,
        root_path=Path(args.data_path), ext=args.ext, logger=logger
    )
    logger.info(f'Total number of samples: \t{len(demo_dataset)}')

    model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=demo_dataset)
    model.load_params_from_file(filename=args.ckpt, logger=logger, to_cpu=True)
    model.cuda()
    model.eval()

    detection_result = []
    time_collate = []
    time_load = []
    time_inference = []

    cap = cv2.VideoCapture('/content/drive/MyDrive/Lidar Data/video/cameraCapture-230718171927_5min_600RPM_cut.mp4')
    out = cv2.VideoWriter('../2023-07-18-17-19-31_5min_600RPM.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 15, (832, 832))
    tracker = MultiObjectTracker(dt=1./15.0)

    with torch.no_grad():
        for idx, data_dict in enumerate(demo_dataset):
            logger.info(f'Visualized sample index: \t{idx + 1}')

            ret, frame = cap.read()

            collate_start = time.time()
            data_dict = demo_dataset.collate_batch([data_dict])
            collate_end = time.time()
            time_collate.append(collate_end - collate_start)

            load_start = time.time()
            load_data_to_gpu(data_dict)
            load_end = time.time()
            time_load.append(load_end - load_start)

            inference_start = time.time()
            pred_dicts, _ = model.forward(data_dict)
            inference_end = time.time()
            time_inference.append(inference_end - inference_start)
            print(pred_dicts[0]['pred_boxes'].shape)
            # V.draw_scenes(
            #     points=data_dict['points'][:, 1:], ref_boxes=pred_dicts[0]['pred_boxes'],
            #     ref_scores=pred_dicts[0]['pred_scores'], ref_labels=pred_dicts[0]['pred_labels']
            # )

            points=data_dict['points'][:, 1:4].cpu()
            rotated_np_points = np.array(rotation_matrix(0, 0, 135).dot(points.T).T)
            point_cloud = [None] * points.shape[0]
            for j in range(len(point_cloud)):
                # point_cloud[i] = (int(140 + 20*rotated_np_points[i][0]), int(730 + 20.5*rotated_np_points[i][1])) # This is for the old dataset
                point_cloud[j] = (int(730 + 20*rotated_np_points[j][0]), int(750 + 20*rotated_np_points[j][1]))
            for j in range(len(point_cloud)):
                frame = cv2.circle(frame, point_cloud[j], radius=0, color=(255, 0, 0), thickness=-1) # circle function is for ploting the point cloud data on the picture frame
            
            np_pre_result = np.array(pred_dicts[0]['pred_boxes'].cpu()) # ['x', 'y', 'z', 'x_size', 'y_size', 'z_size', 'yaw', 'x_velosity', 'y_velosity']
            np_labels = np.array(pred_dicts[0]['pred_labels'].cpu())
            
            np_box_size = np.array(rotation_matrix(0, 0, 135).dot(np_pre_result[:,3:6].T).T)
            rotated_np_pre_result = np.array(rotation_matrix(0, 0, 135).dot(np_pre_result[:,0:3].T).T)
            
            object = [None] * pred_dicts[0]['pred_boxes'].cpu().shape[0]
            pt = [None] * pred_dicts[0]['pred_boxes'].cpu().shape[0]
            for k in range(len(object)):
                # object[i] = (int(140 + 20*rotated_np_pre_result[i][0]), int(730 + 20.5*rotated_np_pre_result[i][1])) # This is for the old dataset
                object[k] = (int(730 + 20*rotated_np_pre_result[k][0]), int(750 + 20*rotated_np_pre_result[k][1]))

            for object_idx in range(len(object)):
                rotated_np_box_size = rotation_matrix(0, 0, np_pre_result[object_idx,6]/np.pi*180).dot(np_box_size[object_idx].T).T
                pt1 = [int(object[object_idx][0]-20*abs(rotated_np_box_size[0])/2), int(object[object_idx][1]-20*abs(rotated_np_box_size[1])/2)]
                pt2 = [int(object[object_idx][0]+20*abs(rotated_np_box_size[0])/2), int(object[object_idx][1]+20*abs(rotated_np_box_size[1])/2)]
                pt[object_idx] = np.array(pt1 + pt2)

            NMSed_bbox = torch_nms(torch.FloatTensor(np.array(pt)), pred_dicts[0]['pred_scores'].cpu(), 0.1)
            
            detections = []
            tracking_start = time.time()
            for NMSed_item in NMSed_bbox:
                detections.append(Detection(box=np.array(pt[NMSed_item]), score=np.array(pred_dicts[0]['pred_scores'].cpu())[NMSed_item], class_id=np_labels[NMSed_item]))
            tracker.step(detections=detections)
            tracks = tracker.active_tracks()
            tracking_end = time.time()

            for track in tracks:
                box = np.int64(track.box)
                if track.class_id == 1 or track.class_id == 2 or track.class_id == 3 or track.class_id == 4 or track.class_id == 5 :
                    frame = cv2.rectangle(frame, box[:2], box[2:], (0, 255, 0), thickness=2)
                if track.class_id == 7 or track.class_id == 8 or track.class_id == 9:
                    frame = cv2.rectangle(frame, box[:2], box[2:], (0, 0, 255), thickness=2)
            
            # saving_result = np.append(saving_result, {'frame_id': i, 'labels': [tracks[k][3] for k in range(len(tracks))], 'boxes': [[(int(tracks[k][1][0]), int(tracks[k][1][1])), ((int(tracks[k][1][2]), int(tracks[k][1][3])))] for k in range(len(tracks))], 'scores': [tracks[k][2] for k in range(len(tracks))]})

    # np.save('../detection_result.npy', detection_result, allow_pickle=True)


    print("data collate time per frame:", np.mean(time_collate))
    print("data load time per frame:", np.mean(time_load))
    print("inference time per frame:", np.mean(time_inference))
    logger.info('Demo done.')
    cv2.imwrite('../frame.jpg', frame)

if __name__ == '__main__':
    main()
