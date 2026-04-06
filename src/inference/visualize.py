"""
BEVFusion Visualization Script
This script provides BEV (Bird's Eye View) visualization for detection results.

Requirements:
    - numpy
    - matplotlib
    - pyquaternion
    - nuscenes-devkit
    - imageio (for creating GIFs)

Usage:
    python visualize.py --show
"""

import numpy as np
import json
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from pyquaternion import Quaternion
from nuscenes.nuscenes import NuScenes
import os
from nuscenes.utils import splits
import imageio  # 添加imageio用于创建GIF
import time
from nuscenes.prediction.helper import PredictHelper
import argparse  # 添加argparse用于命令行参数

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
            'truck': '#00AA00',         # 深绿色
            'construction_vehicle': '#FF00FF',  # 品红
            'bus': '#FFFF00',           # 黄色
            'trailer': '#FFA500',       # 橙色
            'barrier': '#FF0000',       # 红色
            'motorcycle': '#800080',    # 紫色
            'bicycle': '#008B8B',       # 深青色
            'pedestrian': '#FF69B4',    # 粉色
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
        plt.style.use('default')

        # Draw point cloud
        mask = (points[:, 0] > self.pc_range[0]) & (points[:, 0] < self.pc_range[2]) & \
               (points[:, 1] > self.pc_range[1]) & (points[:, 1] < self.pc_range[3])
        plt.scatter(points[mask, 0], points[mask, 1], s=0.1, c='gray', alpha=0.3)

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
        ax1.scatter(points[mask, 0], points[mask, 1], s=0.1, c='gray', alpha=0.3)
        ax1.set_xlim(self.pc_range[0], self.pc_range[2])
        ax1.set_ylim(self.pc_range[1], self.pc_range[3])
        ax1.set_title('Point Cloud (BEV)')
        ax1.set_xlabel('X (m)')
        ax1.set_ylabel('Y (m)')
        ax1.grid(True, alpha=0.3)

        # Right: Detections
        ax2.scatter(points[mask, 0], points[mask, 1], s=0.1, c='gray', alpha=0.3)

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

    def draw_single_frame(self, sample_token, pred_json_path, score_threshold=0.3, show_gt=True, 
                         show_ego=False):
        """
        Draw a single frame for animation purposes.

        Args:
            sample_token: NuScenes sample token
            pred_json_path: Path to prediction JSON file
            score_threshold: Minimum score to display predictions
            show_gt: Whether to show ground truth boxes
            show_ego: Whether to show ego vehicle position
            
        Returns:
            Matplotlib figure object
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

        fig, ax = plt.subplots(1, 1, figsize=(12, 12))

        # Draw point cloud
        mask = (points[:, 0] > self.pc_range[0]) & (points[:, 0] < self.pc_range[2]) & \
               (points[:, 1] > self.pc_range[1]) & (points[:, 1] < self.pc_range[3])
        ax.scatter(points[mask, 0], points[mask, 1], s=0.1, c='gray', alpha=0.3)

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

                ax.plot(corners[:, 0], corners[:, 1], c='red', linestyle='--',
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

            ax.plot(rect[:, 0], rect[:, 1], c=color, linewidth=2)

            # Add label with score
            label = f"{box['detection_name']}:{box['detection_score']:.2f}"
            ax.text(pos[0], pos[1], label, color=color, fontsize=6,
                    ha='center', va='bottom')

        # Draw ego vehicle if requested
        if show_ego:
            ax.plot(0, 0, 's', color='blue', markersize=10, label='Ego Vehicle')
        
        ax.set_xlim(self.pc_range[0], self.pc_range[2])
        ax.set_ylim(self.pc_range[1], self.pc_range[3])
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_title(f"Sample: {sample_token[:8]}... (Colored: Pred, Red dashed: GT)")
        ax.grid(True, alpha=0.3)
        
        return fig

    def create_scene_animation(self, scene_token, pred_json_path, output_path="scene_animation.gif", 
                              score_threshold=0.3, show_gt=True, fps=2):
        """
        Create an animation of an entire scene over time.

        Args:
            scene_token: NuScenes scene token
            pred_json_path: Path to prediction JSON file
            output_path: Path to save the animation
            score_threshold: Minimum score to display predictions
            show_gt: Whether to show ground truth boxes
            fps: Frames per second for the animation
        """
        scene = self.nusc.get('scene', scene_token)
        first_sample_token = scene['first_sample_token']
        last_sample_token = scene['last_sample_token']
        
        # Get all samples in the scene
        samples = []
        current_token = first_sample_token
        while current_token != '':
            sample = self.nusc.get('sample', current_token)
            samples.append(sample)
            if current_token == last_sample_token:
                break
            current_token = sample['next']
        
        # Create frames for animation
        frames = []
        print(f"Creating animation for scene {scene['name']} ({len(samples)} samples)...")
        
        for i, sample in enumerate(samples):
            print(f"Processing frame {i+1}/{len(samples)}...")
            
            fig = self.draw_single_frame(
                sample['token'], 
                pred_json_path, 
                score_threshold=score_threshold,
                show_gt=show_gt,
                show_ego=True
            )
            
            # Save to temporary buffer
            temp_path = f"temp_frame_{i}.png"
            fig.savefig(temp_path, dpi=100, bbox_inches='tight')
            plt.close(fig)
            
            # Read the image and add to frames
            frame = imageio.imread(temp_path)
            frames.append(frame)
        
        # Create GIF
        print(f"Saving animation to {output_path}...")
        imageio.mimsave(output_path, frames, duration=1/fps)
        
        # Clean up temporary files
        for i in range(len(samples)):
            temp_path = f"temp_frame_{i}.png"
            if os.path.exists(temp_path):
                os.remove(temp_path)
                
        print(f"Animation saved: {output_path}")

    def animate_sample_sequence(self, sample_tokens, pred_json_path, output_path="sequence_animation.gif",
                               score_threshold=0.3, show_gt=True, fps=4):
        """
        Create an animation from a sequence of sample tokens.

        Args:
            sample_tokens: List of sample tokens in chronological order
            pred_json_path: Path to prediction JSON file
            output_path: Path to save the animation
            score_threshold: Minimum score to display predictions
            show_gt: Whether to show ground truth boxes
            fps: Frames per second for the animation
        """
        frames = []
        print(f"Creating animation for {len(sample_tokens)} samples...")
        
        for i, token in enumerate(sample_tokens):
            print(f"Processing frame {i+1}/{len(sample_tokens)}...")
            
            fig = self.draw_single_frame(
                token, 
                pred_json_path, 
                score_threshold=score_threshold,
                show_gt=show_gt,
                show_ego=True
            )
            
            # Save to temporary buffer
            temp_path = f"temp_frame_{i}.png"
            fig.savefig(temp_path, dpi=100, bbox_inches='tight')
            plt.close(fig)
            
            # Read the image and add to frames
            frame = imageio.imread(temp_path)
            frames.append(frame)
        
        # Create GIF
        print(f"Saving animation to {output_path}...")
        imageio.mimsave(output_path, frames, duration=1/fps)
        
        # Clean up temporary files
        for i in range(len(sample_tokens)):
            temp_path = f"temp_frame_{i}.png"
            if os.path.exists(temp_path):
                os.remove(temp_path)
                
        print(f"Animation saved: {output_path}")

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

    def play_scene_animation(self, scene_token, pred_json_path, 
                            score_threshold=0.3, show_gt=True, interval=500):
        """
        Play animation of an entire scene in real-time without window flickering.
        
        Args:
            scene_token: NuScenes scene token
            pred_json_path: Path to prediction JSON file
            score_threshold: Minimum score to display predictions
            show_gt: Whether to show ground truth boxes
            interval: Time between frames in milliseconds
        """
        scene = self.nusc.get('scene', scene_token)
        first_sample_token = scene['first_sample_token']
        last_sample_token = scene['last_sample_token']
        
        # Get all samples in the scene
        samples = []
        current_token = first_sample_token
        while current_token != '':
            sample = self.nusc.get('sample', current_token)
            samples.append(sample)
            if current_token == last_sample_token:
                break
            current_token = sample['next']
        
        print(f"Playing animation for scene {scene['name']} ({len(samples)} samples)...")
        
        # Load predictions once
        with open(pred_json_path, 'r') as f:
            all_preds = json.load(f)['results']
        
        # Create figure and axis once
        self.fig, self.ax = plt.subplots(1, 1, figsize=(12, 12))
        self.current_frame = [0]  # Use list to make it mutable in nested function
        self.samples = samples
        self.all_preds = all_preds
        self.score_threshold = score_threshold
        self.show_gt = show_gt
        
        def update(frame_idx):
            """Update function for animation."""
            self.ax.clear()
            sample = self.samples[frame_idx]
            sample_token = sample['token']
            
            # Get lidar data
            lidar_token = sample['data']['LIDAR_TOP']
            lidar_path = self.nusc.get_sample_data_path(lidar_token)
            
            # Load point cloud
            points = np.fromfile(lidar_path, dtype=np.float32).reshape(-1, 5)[:, :2]
            
            # Get predictions for this sample
            preds = self.all_preds.get(sample_token, [])
            
            # Prepare transformation matrices
            lidar_data = self.nusc.get('sample_data', lidar_token)
            ego_pose = self.nusc.get('ego_pose', lidar_data['ego_pose_token'])
            cs_record = self.nusc.get('calibrated_sensor', lidar_data['calibrated_sensor_token'])
            
            # Draw point cloud
            mask = (points[:, 0] > self.pc_range[0]) & (points[:, 0] < self.pc_range[2]) & \
                   (points[:, 1] > self.pc_range[1]) & (points[:, 1] < self.pc_range[3])
            self.ax.scatter(points[mask, 0], points[mask, 1], s=0.1, c='gray', alpha=0.3)
            
            # Draw ground truth boxes
            if self.show_gt:
                for ann_token in sample['anns']:
                    box = self.nusc.get_box(ann_token)
                    
                    # Transform from Global to Lidar coordinate system
                    box.translate(-np.array(ego_pose['translation']))
                    box.rotate(Quaternion(ego_pose['rotation']).inverse)
                    box.translate(-np.array(cs_record['translation']))
                    box.rotate(Quaternion(cs_record['rotation']).inverse)
                    
                    # Get BEV view bottom 4 corners
                    corners = box.corners()[:2, [0, 1, 5, 4]].T
                    corners = np.vstack([corners, corners[0]])
                    
                    self.ax.plot(corners[:, 0], corners[:, 1], c='red', linestyle='--',
                            linewidth=1.5, alpha=0.6)
            
            # Draw prediction boxes
            for box in preds:
                if box['detection_score'] < self.score_threshold:
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
                
                # Size restoration
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
                
                self.ax.plot(rect[:, 0], rect[:, 1], c=color, linewidth=2)
                
                # Add label with score
                label = f"{box['detection_name']}:{box['detection_score']:.2f}"
                self.ax.text(pos[0], pos[1], label, color=color, fontsize=6,
                        ha='center', va='bottom')
            
            # Draw ego vehicle
            self.ax.plot(0, 0, 's', color='blue', markersize=10, label='Ego Vehicle')
            
            # Set axis properties
            self.ax.set_xlim(self.pc_range[0], self.pc_range[2])
            self.ax.set_ylim(self.pc_range[1], self.pc_range[3])
            self.ax.set_xlabel('X (m)')
            self.ax.set_ylabel('Y (m)')
            self.ax.set_title(f"Frame {frame_idx+1}/{len(self.samples)} - Sample: {sample_token[:8]}... (Colored: Pred, Red dashed: GT)")
            self.ax.grid(True, alpha=0.3)
            
            return []
        
        # Create animation
        self.anim = FuncAnimation(
            self.fig, 
            update, 
            frames=len(samples),
            interval=interval,
            blit=False,
            repeat=True
        )
        
        plt.tight_layout()
        plt.show()

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

    def create_animation_for_validation_scenes(self, pred_json_path, output_dir="animation_results",
                                             max_scenes=3, max_samples_per_scene=20, fps=2):
        """
        Create animations for validation scenes.

        Args:
            pred_json_path: Path to prediction JSON file
            output_dir: Directory to save animations
            max_scenes: Maximum number of scenes to animate
            max_samples_per_scene: Maximum number of samples per scene
            fps: Frames per second for the animation
        """
        os.makedirs(output_dir, exist_ok=True)

        if self.nusc.version == 'v1.0-mini':
            eval_set = 'mini_val'
        else:
            eval_set = 'val'

        val_scenes = splits.create_splits_scenes()[eval_set]

        scene_count = 0
        for scene in self.nusc.scene:
            if scene_count >= max_scenes:
                break
                
            scene_name = scene['name']
            if scene_name not in val_scenes:
                continue
                
            print(f"Animating scene: {scene_name}")
            
            # Get samples in the scene
            samples = []
            current_token = scene['first_sample_token']
            last_token = scene['last_sample_token']
            
            while current_token != '' and len(samples) < max_samples_per_scene:
                sample = self.nusc.get('sample', current_token)
                samples.append(sample)
                if current_token == last_token:
                    break
                current_token = sample['next']
            
            if len(samples) < 2:
                print(f"Skipping {scene_name}, not enough samples for animation")
                continue
            
            # Create animation for this scene
            output_path = os.path.join(output_dir, f"animation_{scene_name}.gif")
            sample_tokens = [s['token'] for s in samples]
            
            self.animate_sample_sequence(
                sample_tokens=sample_tokens,
                pred_json_path=pred_json_path,
                output_path=output_path,
                fps=fps
            )
            
            scene_count += 1

        print(f"\nTotal animated scenes: {scene_count}")
        print(f"Output directory: {output_dir}")

    def play_all_scenes(self, pred_json_path, max_samples_per_scene=20, fps=5, max_scenes=None):
        """
        Play all scenes in the dataset sequentially in real-time animation.

        Args:
            pred_json_path: Path to prediction JSON file
            max_samples_per_scene: Maximum number of samples per scene to animate
            fps: Frames per second for the animation
            max_scenes: Maximum number of scenes to play (None for all scenes)
        """
        if self.nusc.version == 'v1.0-mini':
            eval_set = 'mini_val'
        else:
            eval_set = 'val'

        val_scenes = splits.create_splits_scenes()[eval_set]

        print(f"Playing animations for all scenes in {eval_set} set...")
        
        # Load predictions once
        with open(pred_json_path, 'r') as f:
            all_preds = json.load(f)['results']
        
        # Collect all samples from all scenes
        all_samples = []
        scene_boundaries = []  # Track where each scene starts
        scene_count = 0  # Track number of scenes added
        
        for scene_idx, scene in enumerate(self.nusc.scene):
            scene_name = scene['name']
            if scene_name not in val_scenes:
                continue
                
            # Check if we've reached the maximum number of scenes
            if max_scenes is not None and scene_count >= max_scenes:
                print(f"\nReached maximum scenes limit: {max_scenes}")
                break
                
            print(f"Loading scene: {scene_name}")
            
            # Get samples in the scene
            samples = []
            current_token = scene['first_sample_token']
            last_token = scene['last_sample_token']
            
            while current_token != '' and len(samples) < max_samples_per_scene:
                sample = self.nusc.get('sample', current_token)
                samples.append({
                    'sample': sample,
                    'scene_name': scene_name
                })
                if current_token == last_token:
                    break
                current_token = sample['next']
            
            if len(samples) < 2:
                print(f"Skipping {scene_name}, not enough samples for animation")
                continue
            
            scene_boundaries.append(len(all_samples))
            all_samples.extend(samples)
            scene_count += 1
        
        if len(all_samples) == 0:
            print("No valid scenes to play!")
            return
        
        print(f"\nTotal frames to play: {len(all_samples)} from {len(scene_boundaries)} scenes")
        
        # Create single figure and axis for all scenes
        self.fig, self.ax = plt.subplots(1, 1, figsize=(12, 12))
        self.all_samples = all_samples
        self.all_preds = all_preds
        self.score_threshold = 0.3
        self.show_gt = True
        
        def update(frame_idx):
            """Update function for animation."""
            self.ax.clear()
            
            sample_data = self.all_samples[frame_idx]
            sample = sample_data['sample']
            scene_name = sample_data['scene_name']
            sample_token = sample['token']
            
            # Get lidar data
            lidar_token = sample['data']['LIDAR_TOP']
            lidar_path = self.nusc.get_sample_data_path(lidar_token)
            
            # Load point cloud
            points = np.fromfile(lidar_path, dtype=np.float32).reshape(-1, 5)[:, :2]
            
            # Get predictions for this sample
            preds = self.all_preds.get(sample_token, [])
            
            # Prepare transformation matrices
            lidar_data = self.nusc.get('sample_data', lidar_token)
            ego_pose = self.nusc.get('ego_pose', lidar_data['ego_pose_token'])
            cs_record = self.nusc.get('calibrated_sensor', lidar_data['calibrated_sensor_token'])
            
            # Draw point cloud
            mask = (points[:, 0] > self.pc_range[0]) & (points[:, 0] < self.pc_range[2]) & \
                   (points[:, 1] > self.pc_range[1]) & (points[:, 1] < self.pc_range[3])
            self.ax.scatter(points[mask, 0], points[mask, 1], s=0.1, c='gray', alpha=0.3)
            
            # Draw ground truth boxes
            if self.show_gt:
                for ann_token in sample['anns']:
                    box = self.nusc.get_box(ann_token)
                    
                    # Transform from Global to Lidar coordinate system
                    box.translate(-np.array(ego_pose['translation']))
                    box.rotate(Quaternion(ego_pose['rotation']).inverse)
                    box.translate(-np.array(cs_record['translation']))
                    box.rotate(Quaternion(cs_record['rotation']).inverse)
                    
                    # Get BEV view bottom 4 corners
                    corners = box.corners()[:2, [0, 1, 5, 4]].T
                    corners = np.vstack([corners, corners[0]])
                    
                    self.ax.plot(corners[:, 0], corners[:, 1], c='red', linestyle='--',
                            linewidth=1.5, alpha=0.6)
            
            # Draw prediction boxes
            for box in preds:
                if box['detection_score'] < self.score_threshold:
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
                
                # Size restoration
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
                
                self.ax.plot(rect[:, 0], rect[:, 1], c=color, linewidth=2)
                
                # Add label with score
                label = f"{box['detection_name']}:{box['detection_score']:.2f}"
                self.ax.text(pos[0], pos[1], label, color=color, fontsize=6,
                        ha='center', va='bottom')
            
            # Draw ego vehicle
            self.ax.plot(0, 0, 's', color='blue', markersize=10, label='Ego Vehicle')
            
            # Set axis properties
            self.ax.set_xlim(self.pc_range[0], self.pc_range[2])
            self.ax.set_ylim(self.pc_range[1], self.pc_range[3])
            self.ax.set_xlabel('X (m)')
            self.ax.set_ylabel('Y (m)')
            self.ax.set_title(f"Scene: {scene_name} - Frame {frame_idx+1}/{len(self.all_samples)}")
            self.ax.grid(True, alpha=0.3)
            
            return []
        
        # Create animation for all scenes
        interval = int(1000 / fps)  # Convert fps to milliseconds
        self.anim = FuncAnimation(
            self.fig, 
            update, 
            frames=len(all_samples),
            interval=interval,
            blit=False,
            repeat=True
        )
        
        plt.tight_layout()
        plt.show()
                
        print(f"\nFinished playing all scenes!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='BEVFusion Visualization with Animation Support')
    parser.add_argument('--show', action='store_true', help='Automatically play all scenes in animation')
    parser.add_argument('--data-root', type=str, default=None, help='Path to NuScenes dataset')
    parser.add_argument('--pred-path', type=str, default='results_nusc.json', help='Path to prediction JSON file')
    parser.add_argument('--max-samples', type=int, default=20, help='Maximum samples per scene to animate')
    parser.add_argument('--max-scenes', type=int, default=None, help='Maximum number of scenes to play (default: all scenes)')
    parser.add_argument('--fps', type=int, default=5, help='Frames per second for animation')
    
    args = parser.parse_args()

    # Get project root directory
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    # Set data root path (use absolute path)
    dataroot = args.data_root or os.path.join(project_root, "data/nuscenes-mini")

    # Create visualizer
    vis = NuScenesVisualizer(dataroot=dataroot, version='v1.0-mini')

    # Check if results file exists
    results_file = args.pred_path
    if not os.path.exists(results_file):
        print(f"Error: Results file not found: {results_file}")
        print("Please run bevfusion_evaluator.py first to generate predictions.")
        import sys
        sys.exit(1)

    if args.show:
        # Automatically play all scenes
        print("Starting animation playback for all scenes...")
        vis.play_all_scenes(
            pred_json_path=results_file,
            max_samples_per_scene=args.max_samples,
            max_scenes=args.max_scenes,
            fps=args.fps
        )
    else:
        # Create animations for validation scenes
        print("Generating BEV animations...")
        vis.create_animation_for_validation_scenes(
            pred_json_path=results_file, 
            output_dir='animation_results', 
            max_scenes=2, 
            max_samples_per_scene=15, 
            fps=2
        )

        print("\nAnimation creation complete!")