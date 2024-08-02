#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# flake8: noqa
#
# Copyright 2023-2024 Herman Ye @Auromix
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from dex_retargeting.retargeting_config import RetargetingConfig
from auro_utils.loggers.logger import Logger
from auro_utils.io.file_operator import get_project_top_level_dir
import cv2
import os
import toml
import numpy as np
from auro_puppeteer.human_hand_processor.single_hand_detector import SingleHandDetector


RIGHT_HAND_AXES = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
DETECTOR_WRIST_ROT_TO_HAND_WRIST_ROT = np.array([[0, 0, 1], [0, 1, 0], [-1, 0, 0]])


class RGBCameraProcessor:
    def __init__(self, config):
        # Get singleton logger
        self.logger = Logger()
        # Init variables
        self.config = config
        self.left_hand_config = config["left_hand"]
        self.right_hand_config = config["right_hand"]
        self.left_hand_dex_retargeting = None
        self.right_hand_dex_retargeting = None
        self.left_hand_detector = None
        self.right_hand_detector = None

        # Init Left hand
        if self.left_hand_config["enabled"]:
            left_hand_urdf_dir = os.path.join(
                get_project_top_level_dir(), self.left_hand_config["urdf_dir"]
            )
            left_hand_dex_retargeting_config_path = os.path.join(
                get_project_top_level_dir(),
                self.left_hand_config["dex_retargeting_config_path"],
            )

            RetargetingConfig.set_default_urdf_dir(left_hand_urdf_dir)
            self.left_hand_dex_retargeting_config = RetargetingConfig.load_from_file(
                left_hand_dex_retargeting_config_path
            )
            self.left_hand_dex_retargeting = (
                self.left_hand_dex_retargeting_config.build()
            )
            self.left_hand_detector = SingleHandDetector(
                hand_type="Left",
                selfie=self.left_hand_config["selfie"],
            )
            self.logger.log_success("RGBCameraProcessor: Left hand initialized.")
        # Init Right hand
        if self.right_hand_config["enabled"]:
            right_hand_urdf_dir = os.path.join(
                get_project_top_level_dir(), self.right_hand_config["urdf_dir"]
            )
            right_hand_dex_retargeting_config_path = os.path.join(
                get_project_top_level_dir(),
                self.right_hand_config["dex_retargeting_config_path"],
            )

            RetargetingConfig.set_default_urdf_dir(right_hand_urdf_dir)
            self.right_hand_dex_retargeting_config = RetargetingConfig.load_from_file(
                right_hand_dex_retargeting_config_path
            )
            self.right_hand_dex_retargeting = (
                self.right_hand_dex_retargeting_config.build()
            )
            self.right_hand_detector = SingleHandDetector(
                hand_type="Right",
                selfie=self.right_hand_config["selfie"],
            )
            self.logger.log_success("RGBCameraProcessor: Right hand initialized.")


class AppleVisionProProcessor:
    def __init__(self, config):
        # Get singleton logger
        self.logger = Logger()
        # TODO@Herman


class HumanHandProcessor:
    def __init__(self, config):
        self.right_hand_status = None
        self.left_hand_status = None
        # Get project top level dir
        self.project_top_level_dir = get_project_top_level_dir()
        config_path = os.path.join(self.project_top_level_dir, config)
        # Load config
        self.config = toml.load(config_path)
        # Init logger
        self.logger = Logger(log_level=self.config["logger_config"]["log_level"])

        # Init Human Hand Processor
        self.human_hand_processor_config = self.config["human_hand_processor"]
        config_str = "\n".join(
            f"{key}:{value}\n"
            for key, value in self.human_hand_processor_config.items()
        )
        self.logger.log_info(
            f"Initializing Human Hand Processor with config: \n{config_str}"
        )
        self.input_device = self.human_hand_processor_config["input_device"]
        assert self.input_device in [
            "rgb",
            "apple_vision_pro",
        ], f"Invalid input device: {self.input_device}"
        if self.input_device == "rgb":
            self.processor = RGBCameraProcessor(self.human_hand_processor_config)
        elif self.input_device == "apple_vision_pro":
            self.processor = AppleVisionProProcessor(self.human_hand_processor_config)

    def get_joint_indexes_for_dex_retargeting_in_isaac_sim(self, hand, robot):
        """
        Get the joint indexes for dex retargeting in isaac sim

        Args:
            hand (str): hand name, either "left_hand" or "right_hand"
            robot (Robot): isaac sim robot instance

        Returns:
            List[int]: joint indexes for dex retargeting in isaac sim
        """
        # Get active joint names in isaac_sim
        isaac_sim_active_joint_names = robot.dof_names
        # Get empty common joint names
        common_joint_names = []
        if hand == "left_hand":
            # Get dex retargeting joint names
            left_dex_retargeting_joint_names = self.processor.left_hand_dex_retargeting.joint_names
            
            for name in left_dex_retargeting_joint_names:
                if name in isaac_sim_active_joint_names:
                    common_joint_names.append(name)
                    
        elif hand == "right_hand":
            # Get dex retargeting joint names
            right_dex_retargeting_joint_names = self.processor.right_hand_dex_retargeting.joint_names
            
            for name in right_dex_retargeting_joint_names:
                if name in isaac_sim_active_joint_names:
                    common_joint_names.append(name)
        else:
            raise ValueError("Invalid hand type. Must be 'left_hand' or 'right_hand'.")
        # Get common joint indexes
        common_joint_indexes = [robot.get_dof_index(x) for x in common_joint_names]

        return common_joint_indexes

    def detect_left_hand(self, rgb):
        """
        Detects the left hand joints from an RGB image and computes the joint angles and wrist rotation.

        Args:
            rgb: The RGB image input used for hand detection.

        Returns:
            tuple: A tuple containing:
                - hand_joint_angles: The angles of the joints of the left hand.
                - wrist_rot: The rotation of the wrist.

        Raises:
            ValueError: If the input device specified in human_hand_processor_config is not 'rgb'.
            SystemExit: If the left hand detector is not initialized.

        """
        if self.human_hand_processor_config["input_device"] == "rgb":
            retargeting_type = (
                self.processor.left_hand_dex_retargeting.optimizer.retargeting_type
            )

            if self.processor.left_hand_detector is None:
                self.logger.log_warning("left_hand detector is not initialized.")
                exit(1)

            (
                num_of_hand,
                joint_positions_to_wrist,
                key_points_2d,
                wrist_rot,
            ) = self.processor.left_hand_detector.detect(rgb)

            if joint_positions_to_wrist is None:
                self.left_hand_status = None
                self.logger.log_debug("left_hand is not detected.")
                return None, None
            else:

                self.left_hand_status = {
                    "num_of_hand": num_of_hand,
                    "joint_positions_to_wrist": joint_positions_to_wrist,
                    "key_points_2d": key_points_2d,
                    "wrist_rot": wrist_rot,
                }
                indices = (
                    self.processor.left_hand_dex_retargeting.optimizer.target_link_human_indices
                )
                self.logger.log_debug(f"Target link human indices: {indices}")
                if retargeting_type == "POSITION":
                    ref_value = joint_positions_to_wrist[indices, :]
                else:
                    origin_indices = indices[0, :]
                    task_indices = indices[1, :]
                    ref_value = (
                        joint_positions_to_wrist[task_indices, :]
                        - joint_positions_to_wrist[origin_indices, :]
                    )
                hand_joint_angles = self.processor.left_hand_dex_retargeting.retarget(
                    ref_value
                )
                return hand_joint_angles, wrist_rot
        else:
            raise ValueError(
                f"Unsupported input device: {self.human_hand_processor_config['input_device']}"
            )

    def detect_right_hand(self, rgb):
        """
        Detects the right hand joints from an RGB image and computes the joint angles and wrist rotation.

        Args:
            rgb: The RGB image input used for hand detection.

        Returns:
            tuple: A tuple containing:
                - hand_joint_angles: The angles of the joints of the right hand.
                - wrist_rot: The rotation of the wrist.

        Raises:
            ValueError: If the input device specified in human_hand_processor_config is not 'rgb'.
            SystemExit: If the right hand detector is not initialized.

        """
        if self.human_hand_processor_config["input_device"] == "rgb":
            retargeting_type = (
                self.processor.right_hand_dex_retargeting.optimizer.retargeting_type
            )

            if self.processor.right_hand_detector is None:
                self.logger.log_warning("right_hand detector is not initialized.")
                exit(1)

            (
                num_of_hand,
                joint_positions_to_wrist,
                key_points_2d,
                wrist_rot,
            ) = self.processor.right_hand_detector.detect(rgb)

            if joint_positions_to_wrist is None:
                self.right_hand_status = None
                self.logger.log_debug("right_hand is not detected.")
                return None, None
            else:
                self.right_hand_status = {
                    "num_of_hand": num_of_hand,
                    "joint_positions_to_wrist": joint_positions_to_wrist,
                    "key_points_2d": key_points_2d,
                    "wrist_rot": wrist_rot,
                }
                indices = (
                    self.processor.right_hand_dex_retargeting.optimizer.target_link_human_indices
                )
                self.logger.log_debug(f"Target link human indices: {indices}")
                if retargeting_type == "POSITION":
                    ref_value = joint_positions_to_wrist[indices, :]
                else:
                    origin_indices = indices[0, :]
                    task_indices = indices[1, :]
                    ref_value = (
                        joint_positions_to_wrist[task_indices, :]
                        - joint_positions_to_wrist[origin_indices, :]
                    )
                hand_joint_angles = self.processor.right_hand_dex_retargeting.retarget(
                    ref_value
                )
                return hand_joint_angles, wrist_rot
        else:
            raise ValueError(
                f"Unsupported input device: {self.human_hand_processor_config['input_device']}"
            )

    def draw_skeleton_on_image(self, frame):
        """
        Draws the skeletons of the detected left and right hands on the given image frame.
        Additionally, draws the 3D axes at the wrist positions.

        Args:
            frame (numpy.ndarray): The image frame on which to draw the skeletons and axes.

        Returns:
            numpy.ndarray: The image frame with the drawn skeletons and axes.

        Notes:
            - The function first checks if the left and right hand detectors are available and if the hand status is not None.
            - If available, it draws the skeleton of the left hand and the 3D axes at the left wrist position.
            - Similarly, it draws the skeleton of the right hand and the 3D axes at the right wrist position.
            - The axes are colored as follows: X-axis in red, Y-axis in green, and Z-axis in blue.
        """
        height, width, _ = frame.shape
        # Draw left hand
        if (
            self.processor.left_hand_detector is not None
            and self.left_hand_status is not None
        ):
            self.processor.left_hand_detector.draw_skeleton_on_image(
                frame, self.left_hand_status["key_points_2d"]
            )
            left_wrist_landmark = self.left_hand_status["key_points_2d"].landmark[0]
            left_wrist_x_2d = int(left_wrist_landmark.x * width)
            left_wrist_y_2d = int(left_wrist_landmark.y * height)
            origin = (left_wrist_x_2d, left_wrist_y_2d)  # x right, y down

            axis_length = 40
            x3D = np.array([axis_length, 0, 0])
            y3D = np.array([0, axis_length, 0])
            z3D = np.array([0, 0, -axis_length])
            x3D = np.dot(self.left_hand_status["wrist_rot"], x3D)
            y3D = np.dot(self.left_hand_status["wrist_rot"], y3D)
            z3D = np.dot(self.left_hand_status["wrist_rot"], z3D)

            # Get 2d end points of the axis
            x_axis_end = self.project_point(x3D, origin)
            y_axis_end = self.project_point(y3D, origin)
            z_axis_end = self.project_point(z3D, origin)

            # Draw axes
            cv2.arrowedLine(frame, origin, x_axis_end, (0, 0, 255), 2, tipLength=0.05)
            cv2.arrowedLine(frame, origin, y_axis_end, (0, 255, 0), 2, tipLength=0.05)
            cv2.arrowedLine(frame, origin, z_axis_end, (255, 0, 0), 2, tipLength=0.05)

        # Draw right hand
        if (
            self.processor.right_hand_detector is not None
            and self.right_hand_status is not None
        ):
            self.processor.right_hand_detector.draw_skeleton_on_image(
                frame, self.right_hand_status["key_points_2d"]
            )
            right_wrist_landmark = self.right_hand_status["key_points_2d"].landmark[0]
            right_wrist_x_2d = int(right_wrist_landmark.x * width)
            right_wrist_y_2d = int(right_wrist_landmark.y * height)
            origin = (right_wrist_x_2d, right_wrist_y_2d)  # x right, y down

            axis_length = 40
            x3D = np.array([axis_length, 0, 0])
            y3D = np.array([0, axis_length, 0])
            z3D = np.array([0, 0, -axis_length])
            x3D = np.dot(self.right_hand_status["wrist_rot"], x3D)
            y3D = np.dot(self.right_hand_status["wrist_rot"], y3D)
            z3D = np.dot(self.right_hand_status["wrist_rot"], z3D)

            # Get 2d end points of the axis
            x_axis_end = self.project_point(x3D, origin)
            y_axis_end = self.project_point(y3D, origin)
            z_axis_end = self.project_point(z3D, origin)

            # Draw axes
            cv2.arrowedLine(frame, origin, x_axis_end, (0, 0, 255), 2, tipLength=0.05)
            cv2.arrowedLine(frame, origin, y_axis_end, (0, 255, 0), 2, tipLength=0.05)
            cv2.arrowedLine(frame, origin, z_axis_end, (255, 0, 0), 2, tipLength=0.05)
        return frame

    def project_point(self, point3D, origin):
        projection_matrix = np.array([[1, 0, 0], [0, 1, 0]])
        point2D = np.dot(projection_matrix, point3D)
        return (int(point2D[0] + origin[0]), int(point2D[1] + origin[1]))


def main():
    # Modify the cap_path variable to point to your video file
    cap_path = 0
    human_hand_processor = HumanHandProcessor(
        config="config/rgb_camera_dual_hand_teleoperation.toml"
    )
    # Init camera
    if cap_path is None:
        cap_path = 0
    cap = cv2.VideoCapture(cap_path)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            continue
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # Detect
        right_hand_joint_angles, right_wrist_rotation = (
            human_hand_processor.detect_right_hand(image)
        )
        left_hand_joint_angles, left_wrist_rotation = (
            human_hand_processor.detect_left_hand(image)
        )
        human_hand_processor.logger.log_info(
            f"Right hand joint angles: {right_hand_joint_angles}\n Right wrist rotation: {right_wrist_rotation}"
        )
        human_hand_processor.logger.log_info(
            f"Left hand joint angles: {left_hand_joint_angles}\n  Left wrist rotation: {left_wrist_rotation}"
        )
        # Draw skeleton on image
        frame = human_hand_processor.draw_skeleton_on_image(frame)

        cv2.imshow("frame", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
