"""
BEVFusion Visualization Script
This script provides BEV (Bird's Eye View) visualization for detection results.

Requirements:
    - numpy
    - matplotlib
    - pyquaternion
    - nuscenes-devkit

Usage:
    python visualize.py
"""

import numpy as np
import json
import matplotlib.pyplot as plt
from pyquaternion import Quaternion
from nuscenes.nuscenes import NuScenes
import os
from nuscenes.utils import splits


class NuScenesVisualizer:
    """BEV visualization tool for NuScenes detection results."""

    def __init__(self, dataroot='data/nuscenes-mini', version='v1.0-mini'):
        """
        Initialize the visualizer.

        Args:
            dataroot: Path to NuScenes dataset
            version: Dataset version
        """
        self.nusc = NuScenes(version=version, dataroot=dataroot, verbose=False)
        # BEVFusion参数
        self.pc_range = [-54.0, -54.0, 54.0, 54.0]

        # 类别颜色映射
        self.class_colors = {
            'car': '#00FF00',           # 绿色
            'truck': '#00FF00',         # 青色
            'construction_vehicle': '#00FF00',  # 品红
            'bus': '#00FF00',           # 黄色
            'trailer': '#00FF00',       # 橙色
            'barrier': '#00FF00',       # 红色
            'motorcycle': '#00FF00',    # 紫色
            'bicycle': '#00FF00',       # 深青色
            'pedestrian': '#00FF00',    # 粉色
            'traffic_cone': '#A52A2A'   # 棕色
        }

    def draw_bev(self, sample_token, pred_json_path, save_path="vis_bev.png",
                 score_threshold=0.3, show_gt=True):
        """
        Draw BEV visualization for a single sample.

        Args:
            sample_token: NuScenes sample token
            pred_json_path: Path to prediction JSON file
            save_path: Path to save the visualization
            score_threshold: Minimum score to display predictions
            show_gt: Whether to show ground truth boxes
        """
        sample = self.nusc.get('sample', sample_token)
        lidar_token = sample['data']['LIDAR_TOP']
        lidar_path = self.nusc.get_sample_data_path(lidar_token)

        # 1. Load point cloud (take first two columns x, y)
        points = np.fromfile(lidar_path, dtype=np.float32).reshape(-1, 5)[:, :2]

        # 2. Load predictions
        with open(pred_json_path, 'r') as f:
            all_preds = json.load(f)['results']
        preds = all_preds.get(sample_token, [])

        # 3. Prepare transformation matrices
        lidar_data = self.nusc.get('sample_data', lidar_token)
        ego_pose = self.nusc.get('ego_pose', lidar_data['ego_pose_token'])
        cs_record = self.nusc.get('calibrated_sensor', lidar_data['calibrated_sensor_token'])

        plt.figure(figsize=(12, 12))
        plt.style.use('dark_background')

        # Draw point cloud
        mask = (points[:, 0] > self.pc_range[0]) & (points[:, 0] < self.pc_range[2]) & \
               (points[:, 1] > self.pc_range[1]) & (points[:, 1] < self.pc_range[3])
        plt.scatter(points[mask, 0], points[mask, 1], s=0.1, c='white', alpha=0.3)

        # 4. Draw ground truth boxes (red dashed line)
        if show_gt:
            for ann_token in sample['anns']:
                box = self.nusc.get_box(ann_token)

                # Transform from Global to Lidar coordinate system
                # 1. Global -> Ego
                box.translate(-np.array(ego_pose['translation']))
                box.rotate(Quaternion(ego_pose['rotation']).inverse)
                # 2. Ego -> Lidar
                box.translate(-np.array(cs_record['translation']))
                box.rotate(Quaternion(cs_record['rotation']).inverse)

                # Get BEV view bottom 4 corners
                corners = box.corners()[:2, [0, 1, 5, 4]].T
                corners = np.vstack([corners, corners[0]])  # Close the curve

                plt.plot(corners[:, 0], corners[:, 1], c='red', linestyle='--',
                        linewidth=1.5, alpha=0.6)

        # 5. Draw prediction boxes (colored solid line)
        for box in preds:
            if box['detection_score'] < score_threshold:
                continue

            # Global -> Lidar transformation
            pos = np.array(box['translation']) - np.array(ego_pose['translation'])
            pos = np.dot(Quaternion(ego_pose['rotation']).inverse.rotation_matrix, pos)
            pos = pos - np.array(cs_record['translation'])
            pos = np.dot(Quaternion(cs_record['rotation']).inverse.rotation_matrix, pos)

            # Rotation restoration
            combined_rot = Quaternion(box['rotation'])
            rel_rot = Quaternion(cs_record['rotation']).inverse * \
                     Quaternion(ego_pose['rotation']).inverse * combined_rot
            yaw = rel_rot.yaw_pitch_roll[0]

            # Size restoration (w, l, h) -> NuScenes format
            w, l, h = box['size']
            cos_y, sin_y = np.cos(yaw), np.sin(yaw)

            rect = np.array([
                [l / 2, w / 2], [l / 2, -w / 2],
                [-l / 2, -w / 2], [-l / 2, w / 2], [l / 2, w / 2]
            ])
            rot_mat = np.array([[cos_y, -sin_y], [sin_y, cos_y]])
            rect = np.dot(rect, rot_mat.T) + pos[:2]

            # Get color for this class
            color = self.class_colors.get(box['detection_name'], '#00FF00')

            plt.plot(rect[:, 0], rect[:, 1], c=color, linewidth=2)

            # Add label with score
            label = f"{box['detection_name']}:{box['detection_score']:.2f}"
            plt.text(pos[0], pos[1], label, color=color, fontsize=6,
                    ha='center', va='bottom')

        plt.xlim(self.pc_range[0], self.pc_range[2])
        plt.ylim(self.pc_range[1], self.pc_range[3])
        plt.xlabel('X (m)')
        plt.ylabel('Y (m)')
        plt.title(f"Sample: {sample_token[:8]}... (Colored: Pred, Red dashed: GT)")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved: {save_path}")

    def draw_comparison(self, sample_token, pred_json_path, save_path="vis_comparison.png"):
        """
        Draw side-by-side comparison of point cloud and detection results.

        Args:
            sample_token: NuScenes sample token
            pred_json_path: Path to prediction JSON file
            save_path: Path to save the visualization
        """
        sample = self.nusc.get('sample', sample_token)
        lidar_token = sample['data']['LIDAR_TOP']
        lidar_path = self.nusc.get_sample_data_path(lidar_token)

        # Load point cloud
        points = np.fromfile(lidar_path, dtype=np.float32).reshape(-1, 5)

        # Load predictions
        with open(pred_json_path, 'r') as f:
            all_preds = json.load(f)['results']
        preds = all_preds.get(sample_token, [])

        # Create figure with two subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))

        # Left: Point cloud only
        mask = ((points[:, 0] > self.pc_range[0]) & (points[:, 0] < self.pc_range[2]) &
                (points[:, 1] > self.pc_range[1]) & (points[:, 1] < self.pc_range[3]))
        ax1.scatter(points[mask, 0], points[mask, 1], s=0.1, c='white', alpha=0.3)
        ax1.set_xlim(self.pc_range[0], self.pc_range[2])
        ax1.set_ylim(self.pc_range[1], self.pc_range[3])
        ax1.set_title('Point Cloud (BEV)')
        ax1.set_xlabel('X (m)')
        ax1.set_ylabel('Y (m)')
        ax1.grid(True, alpha=0.3)

        # Right: Detections
        ax2.scatter(points[mask, 0], points[mask, 1], s=0.1, c='white', alpha=0.3)

        # Prepare transformation
        lidar_data = self.nusc.get('sample_data', lidar_token)
        ego_pose = self.nusc.get('ego_pose', lidar_data['ego_pose_token'])
        cs_record = self.nusc.get('calibrated_sensor', lidar_data['calibrated_sensor_token'])

        for box in preds:
            if box['detection_score'] < 0.3:
                continue

            # Transform to lidar frame
            pos = np.array(box['translation']) - np.array(ego_pose['translation'])
            pos = np.dot(Quaternion(ego_pose['rotation']).inverse.rotation_matrix, pos)
            pos = pos - np.array(cs_record['translation'])
            pos = np.dot(Quaternion(cs_record['rotation']).inverse.rotation_matrix, pos)

            combined_rot = Quaternion(box['rotation'])
            rel_rot = Quaternion(cs_record['rotation']).inverse * \
                     Quaternion(ego_pose['rotation']).inverse * combined_rot
            yaw = rel_rot.yaw_pitch_roll[0]

            w, l, h = box['size']
            cos_y, sin_y = np.cos(yaw), np.sin(yaw)

            rect = np.array([
                [l / 2, w / 2], [l / 2, -w / 2],
                [-l / 2, -w / 2], [-l / 2, w / 2], [l / 2, w / 2]
            ])
            rot_mat = np.array([[cos_y, -sin_y], [sin_y, cos_y]])
            rect = np.dot(rect, rot_mat.T) + pos[:2]

            color = self.class_colors.get(box['detection_name'], '#00FF00')
            ax2.plot(rect[:, 0], rect[:, 1], c=color, linewidth=2)
            ax2.text(pos[0], pos[1], f"{box['detection_name']}",
                    color=color, fontsize=6, ha='center')

        ax2.set_xlim(self.pc_range[0], self.pc_range[2])
        ax2.set_ylim(self.pc_range[1], self.pc_range[3])
        ax2.set_title(f'Detections (n={len([b for b in preds if b["detection_score"] >= 0.3])})')
        ax2.set_xlabel('X (m)')
        ax2.set_ylabel('Y (m)')
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved: {save_path}")

    def visualize_batch(self, pred_json_path, output_dir="vis_results",
                       max_samples=10, show_gt=True):
        """
        Visualize a batch of samples.

        Args:
            pred_json_path: Path to prediction JSON file
            output_dir: Directory to save visualizations
            max_samples: Maximum number of samples to visualize
            show_gt: Whether to show ground truth boxes
        """
        os.makedirs(output_dir, exist_ok=True)

        if self.nusc.version == 'v1.0-mini':
            eval_set = 'mini_val'
        else:
            eval_set = 'val'

        val_scenes = splits.create_splits_scenes()[eval_set]

        count = 0
        for s in self.nusc.sample:
            scene_name = self.nusc.get('scene', s['scene_token'])['name']
            if scene_name in val_scenes:
                save_path = os.path.join(output_dir, f"bev_val_{count}.png")
                self.draw_bev(s['token'], pred_json_path, save_path=save_path, show_gt=show_gt)
                count += 1
                if count >= max_samples:
                    break

        print(f"\nTotal visualizations saved: {count}")
        print(f"Output directory: {output_dir}")

    def create_summary_plot(self, pred_json_path, output_dir="vis_results"):
        """
        Create a summary plot with statistics.

        Args:
            pred_json_path: Path to prediction JSON file
            output_dir: Directory to save the summary
        """
        os.makedirs(output_dir, exist_ok=True)

        with open(pred_json_path, 'r') as f:
            all_preds = json.load(f)['results']

        # Count detections per class
        class_counts = {name: 0 for name in self.class_colors.keys()}
        score_distribution = []

        for sample_token, preds in all_preds.items():
            for pred in preds:
                if pred['detection_name'] in class_counts:
                    class_counts[pred['detection_name']] += 1
                score_distribution.append(pred['detection_score'])

        # Create figure
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

        # Left: Class distribution
        classes = list(class_counts.keys())
        counts = [class_counts[c] for c in classes]
        colors = [self.class_colors[c] for c in classes]

        ax1.barh(classes, counts, color=colors)
        ax1.set_xlabel('Number of Detections')
        ax1.set_title('Detection Count by Class')
        ax1.grid(True, alpha=0.3, axis='x')

        # Right: Score distribution
        ax2.hist(score_distribution, bins=50, color='steelblue', edgecolor='black', alpha=0.7)
        ax2.set_xlabel('Detection Score')
        ax2.set_ylabel('Frequency')
        ax2.set_title('Detection Score Distribution')
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        save_path = os.path.join(output_dir, "summary_statistics.png")
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved summary: {save_path}")


if __name__ == "__main__":
    # Get project root directory
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    # Set data root path (use absolute path)
    dataroot = os.path.join(project_root, "data/nuscenes-mini")

    # Create visualizer
    vis = NuScenesVisualizer(dataroot=dataroot, version='v1.0-mini')

    # Check if results file exists
    results_file = 'results_nusc.json'
    if not os.path.exists(results_file):
        print(f"Error: Results file not found: {results_file}")
        print("Please run bevfusion_evaluator.py first to generate predictions.")
        sys.exit(1)

    # Visualize batch
    print("Generating BEV visualizations...")
    vis.visualize_batch(results_file, output_dir='vis_results', max_samples=10, show_gt=True)

    # Create summary plot
    print("\nGenerating summary statistics...")
    vis.create_summary_plot(results_file, output_dir='vis_results')

    print("\nVisualization complete!")
