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
import argparse
import os
import numpy as np
from typing import Union, Dict
import json
from auro_utils.io.file_operator import (
    get_project_top_level_dir,
    ensure_path_exists,
    check_file_exists,
    read_toml,
)
from auro_utils.loggers.logger import Logger


class SimCore:

    def __init__(self, config: Union[Dict, str] = "default_config.toml") -> None:
        """
        Initialize SimCore with configuration.

        Args:
            config (Union[str, Dict]): Either a path to the configuration TOML file
                or a dictionary containing the configuration.
        """
        self.isaac_sim_assets_root_path = None
        # Get project top level directory
        self.project_dir = get_project_top_level_dir()
        # Create data save path
        self.data_save_path = os.path.join(self.project_dir, "data")
        ensure_path_exists(self.data_save_path)
        # Load config
        self.config = self.load_config(config)
        # Load logger
        self.logger = self.load_logger(self.config.get("logger_config"))
        # Load simulator
        self.simulator = self.load_simulator(self.config.get("simulator_config"))
        # Load world
        self.world = self.load_world(self.config.get("world_config"))
        # Load environment
        if self.config.get("environment_config") is not None:
            self.environment = self.load_environment(
                self.config.get("environment_config")
            )
        # Load robot group
        if self.config.get("robot_group_config") is not None:
            self.robot_group = self.load_robot_group(
                self.config.get("robot_group_config")
            )
        # Load camera group
        if self.config.get("camera_group_config") is not None:
            self.camera_group = self.load_camera_group(
                self.config.get("camera_group_config")
            )
        self.logger.log_debug(f"Data save path: {self.data_save_path}")
        self.logger.log_success("Sim core initialized.")

    def load_config(self, config: Union[str, Dict]) -> Dict:
        """
        Load configuration from TOML file or dictionary.

        Args:
            config (Union[str, Dict]): Either a path to the configuration TOML file
                or a dictionary containing the configuration.

        Returns:
            dict: Configuration dictionary.
        """
        if isinstance(config, dict):
            # If config is already a dictionary, return it directly
            return config
        else:
            # Read config from TOML file
            file_path = os.path.join(self.project_dir, config)
            return read_toml(file_path)

    def load_logger(self, logger_config: Dict) -> Logger:
        """
        Load logger based on configuration.

        Args:
            logger_config (dict): Logger configuration dictionary.
        """
        log_level = logger_config.get("log_level", "info")
        logger = Logger(log_level=log_level)
        logger.log_success(f"Logger loaded with config: {logger_config}")
        return logger

    def load_simulator(self, simulator_config: Dict):
        """
        Load simulator based on configuration.

        Args:
            simulator_config (dict): Simulator configuration dictionary.

        Returns:
            Simulator: Simulator object.
        """
        # Launch Isaac Sim before any other imports
        from isaacsim import SimulationApp

        # Run Isaac Sim
        simulation_app = SimulationApp(simulator_config)

        self.logger.log_success(f"Isaac Sim loaded with config: {simulator_config}")

        import omni.isaac.nucleus as nucleus_utils

        # Get isaac sim assets folder root path
        self.isaac_sim_assets_root_path = nucleus_utils.get_assets_root_path()
        if self.isaac_sim_assets_root_path is None:
            self.logger.log_error("Could not find nucleus server with '/Isaac' folder")
            raise FileNotFoundError("Could not find Isaac Sim assets folder")
        else:
            self.logger.log_info(
                f"Isaac sim assets root path: {self.isaac_sim_assets_root_path}"
            )
        return simulation_app

    def load_world(self, world_config: Dict):
        """
        Loads world based on configuration.

        Args:
            world_config (dict): World configuration dictionary.

        Returns:
            World: World object.
        """

        import omni.isaac.core.utils.prims as prims_utils
        from omni.isaac.core import World

        world = World(stage_units_in_meters=1.0)
        # Set world as default prim
        world_prim = prims_utils.define_prim(prim_path="/World")
        world.stage.SetDefaultPrim(world_prim)

        if world_config.get("add_default_ground_plane", False):
            # Add a ground plane to the stage and register it in the scene
            world.scene.add_default_ground_plane()
        # Reset
        world.reset()
        self.logger.log_success(f"Isaac Sim world loaded with config: {world_config}")
        return world

    def load_environment(self, environment_config: Dict):
        """
        Load environment based on configuration.

        Args:
            environment_config (dict): Environment configuration dictionary.

        Returns:
            Environment: Dictionary of environment objects.
        """

        # import omni.isaac.core.utils.stage as stage_utils
        # import omni.isaac.nucleus as nucleus_utils
        # from omni.isaac.core.prims.xform_prim import XFormPrim

        # Get environment name and file path from configuration
        environment_name = environment_config.get("name")

        detailed_environment_config = environment_config.get("assets", None)
        self.logger.log_debug(
            f"Detailed environment config: {detailed_environment_config}"
        )
        if not detailed_environment_config:
            return None

        environment = {}
        for object_name, object_config in detailed_environment_config.items():
            self.logger.log_debug(
                f"Loading object [{object_name}] in {environment_name} with config: {object_config}"
            )
            environment[object_name] = self.load_object(object_config)
            # xform_prim=XFormPrim(prim_path=object_config.get("prim_path"))

        self.logger.log_success(
            f"Environment [{environment_name}] loaded with objects: {[key for key in environment]}"
        )

        return environment

    def load_object(self, object_config: Dict):
        """
        Load object in Isaac Sim base on the object configuration

        Args:
            object_config (Dict): Object configuration

        Returns:
            object_prim: Loaded object prim in Isaac Sim
        """
        import omni.isaac.nucleus as nucleus_utils
        import omni.isaac.core.utils.stage as stage_utils
        import omni.isaac.core.utils.prims as prim_utils
        import omni.isaac.core.utils.prims as prim_utils

        # Get USD from Isaac Sim assets
        if object_config.get("get_from_isaac_sim_assets"):
            if self.isaac_sim_assets_root_path is not None:
                file_path = self.isaac_sim_assets_root_path + object_config.get(
                    key="file_path"
                )
                if not nucleus_utils.is_file(path=file_path):
                    self.logger.log_error(
                        f"Could not find object USD file at {file_path}"
                    )
                    raise FileNotFoundError(
                        f"Could not find object USD file at {file_path}"
                    )
            else:
                raise FileNotFoundError("Isaac Sim assets root path is not set")
        # Get USD from local user path relative to the project top path
        else:
            file_path = os.path.join(
                self.project_dir,
                object_config.get("file_path"),
            )
        # Check if file exists
        check_file_exists(file_path)

        # Get prim path
        prim_path = object_config.get("prim_path")

        # Create prim and add object prim to the stage
        prim = prim_utils.create_prim(
            prim_path=prim_path,
            usd_path=file_path,
            position=object_config.get("position", None),
            orientation=object_config.get("orientation", None),
            translation=object_config.get("translation", None),
            scale=object_config.get("scale", None),
            semantic_label=object_config.get("semantic_label", None),
            semantic_type=object_config.get("semantic_type", "class"),
            attributes=object_config.get("attributes", None),
        )
        return prim

    def load_robot_group(self, robot_group_config: Dict):
        """
        Load robot_group based on configuration.

        Args:
            robot_group_config (dict): Robot_group configuration dictionary.

        Returns:
            robot_group: Dictionary of robot_group objects.
        """
        robot_group_name = robot_group_config.get("name")

        detailed_robot_group_config = robot_group_config.get("assets", None)
        self.logger.log_debug(
            f"Detailed robot_group config: {detailed_robot_group_config}"
        )

        if not detailed_robot_group_config:
            return None

        robot_group = {}
        for robot_name, robot_config in detailed_robot_group_config.items():
            self.logger.log_debug(
                f"Loading robot [{robot_name}] in {robot_group_name} with config: {robot_config}"
            )
            robot_group[robot_name] = self.load_robot(robot_config)

        self.logger.log_success(
            f"Robot group [{robot_group_name}] loaded with robots: {[key for key in robot_group]}"
        )

        return robot_group

    def load_robot(self, robot_config: Dict):
        """
        Load robot from robot configuration.

        Args:
            robot_config (dict): Robot configuration dictionary.
        Return:
            robot (Robot): Loaded robot instance in Isaac Sim.
        """

        from omni.isaac.core.robots import Robot
        import omni.isaac.nucleus as nucleus_utils
        import omni.isaac.core.utils.stage as stage_utils
        import omni.isaac.core.utils.prims as prim_utils

        # Get USD from Isaac Sim assets
        if robot_config.get("get_from_isaac_sim_assets"):
            if self.isaac_sim_assets_root_path is not None:
                file_path = self.isaac_sim_assets_root_path + robot_config.get(
                    key="file_path"
                )
                if not nucleus_utils.is_file(path=file_path):
                    self.logger.log_error(
                        f"Could not find robot USD file at {file_path}"
                    )
                    raise FileNotFoundError(
                        f"Could not find robot USD file at {file_path}"
                    )
            else:
                raise FileNotFoundError("Isaac Sim assets root path is not set")
        # Get USD from local user path relative to the project top path
        else:
            file_path = os.path.join(
                self.project_dir,
                robot_config.get("file_path"),
            )
        # Check if file exists
        check_file_exists(file_path)

        # Get prim path
        prim_path = robot_config.get("prim_path")

        robot_name_in_scene = robot_config.get("name_in_scene", "default_robot")

        # Add robot asset reference to stage
        # This will create a new XFormPrim and point it to the usd file as a reference
        prim = prim_utils.create_prim(
            prim_path=prim_path,
            usd_path=file_path,
            position=robot_config.get("position", None),
            orientation=robot_config.get("orientation", None),
            translation=robot_config.get("translation", None),
            scale=robot_config.get("scale", None),
            semantic_label=robot_config.get("semantic_label", None),
            semantic_type=robot_config.get("semantic_type", "class"),
            attributes=robot_config.get("attributes", None),
        )
        # stage_utils.add_reference_to_stage(usd_path=file_path, prim_path=prim_path)

        # Wrap the root of robot prim under a Robot(Articulation) class
        # to use high level api to set/ get attributes as well as initializing physics handles needed
        robot = Robot(
            prim_path=prim_path,
            name=robot_name_in_scene,
            position=robot_config.get("position"),
            orientation=robot_config.get("orientation"),
        )

        # Add robot to the scene
        self.world.scene.add(robot)

        # Reset the world
        self.world.reset()
        # Initialize the robot
        robot.initialize()
        # # Get controller
        # robot_articulation_controller = robot.get_articulation_controller()
        return robot

    def load_camera_group(self, camera_group_config: dict) -> None:
        """
        Load the camera group.

        Args:
            camera_group_config (dict): The camera group configuration.

        Returns:
            camera_group: Dictionary of camera_group objects.
        """
        camera_group_name = camera_group_config.get("name")
        detailed_camera_group_config = camera_group_config.get("assets", None)
        self.logger.log_debug(
            f"Detailed camera_group config: {detailed_camera_group_config}"
        )
        if not detailed_camera_group_config:
            return None

        camera_group = {}
        for camera_name, camera_config in detailed_camera_group_config.items():
            self.logger.log_debug(
                f"Loading camera [{camera_name}] in {camera_group_name} with config: {camera_config}"
            )
            camera_group[camera_name] = self.load_camera(camera_config)

        self.logger.log_success(
            f"camera group [{camera_group_name}] loaded with cameras: {[key for key in camera_group]}"
        )

        return camera_group

    def load_camera(self, camera_config: dict) -> None:
        """
        Load camera from camera configuration.

        Args:
            camera_config (dict): camera configuration dictionary.
        Return:
            camera (camera): Loaded camera instance in Isaac Sim.
        """
        from omni.isaac.sensor import Camera
        from omni.isaac.core.utils.viewports import (
            create_viewport_for_camera,
            set_intrinsics_matrix,
        )
        import omni.isaac.core.utils.numpy.rotations as rot_utils

        width, height = camera_config.get("width"), camera_config.get("height")
        fx = camera_config.get("fx")
        fy = camera_config.get("fy")
        cx = camera_config.get("cx")
        cy = camera_config.get("cy")
        pixel_size = camera_config.get("pixel_size")
        # f-number, the ratio of the lens focal length to the diameter of the entrance pupil
        f_stop = camera_config.get("f_stop")
        # in meters, the distance from the camera to the object plane
        focus_distance = camera_config.get("focus_distance")
        # in degrees, the diagonal field of view to be rendered
        diagonal_fov = camera_config.get("diagonal_fov")
        # kannala brandt generic distortion model coefficients (k1, k2, k3, k4)
        distortion_coefficients = camera_config.get("distortion_coefficients")
        projection_type = camera_config.get("projection_type")

        position_offset= camera_config.get("position_offset",np.array([0,0,0]))
        camera = Camera(
            name=camera_config.get("name_in_scene"),
            prim_path=camera_config.get("prim_path"),
            position=camera_config.get("position")+position_offset,
            frequency=camera_config.get("frequency"),
            resolution=(camera_config.get("width"), camera_config.get("height")),
            orientation=None,
        )
        # Set orientation in usd coordinate system if copy from usd
        camera.set_world_pose(
            orientation=camera_config.get("orientation"), camera_axes="usd"
        )
        # Add to scene
        self.world.scene.add(camera)
        # Reset world and initialize camera
        self.world.reset()
        camera.initialize()

        horizontal_aperture = pixel_size * width
        vertical_aperture = pixel_size * height
        focal_length_x = fx * pixel_size
        focal_length_y = fy * pixel_size
        focal_length = (focal_length_x + focal_length_y) / 2  # in mm

        # Set the camera parameters, note the unit conversion between Isaac Sim sensor and Kit

        camera.set_focal_length(focal_length / 10.0)
        camera.set_focus_distance(focus_distance)
        camera.set_lens_aperture(f_stop * 100.0)
        camera.set_horizontal_aperture(horizontal_aperture / 10.0)
        camera.set_vertical_aperture(vertical_aperture / 10.0)

        camera.set_clipping_range(0.02, 1.0e5)

        camera.set_projection_type(projection_type)
        camera.set_kannala_brandt_properties(
            width, height, cx, cy, diagonal_fov, distortion_coefficients
        )

        # Create viewport if specified
        view_port_name = camera_config.get("viewport_name", None)
        if view_port_name:
            create_viewport_for_camera(
                viewport_name=view_port_name,
                camera_prim_path=camera_config.get("prim_path"),
            )

        return camera

    def run_display(self) -> None:
        """
        Run the sim core display test.

        This method continuously runs while the simulation app is active,
        stepping through the world and rendering it as specified.

        Note:
            This method is not necessary if specific tasks are handled separately.
        """
        while self.simulator.is_running():
            # Step
            self.world.step(render=True)
        self.simulator.close()

    def set_robot_joint_angles(self, joint_angles, joint_indexes, robot):
        """
        Set the joint angles of the robot.

        Args:
            joint_angles (List[float]): The joint angles to set.
            joint_indexes (List[int]): The joint indexes to set.
            robot (Robot): The robot to set the joint angles of.
        """

        from omni.isaac.core.utils.types import ArticulationAction

        action = ArticulationAction(
            joint_positions=joint_angles, joint_indices=joint_indexes
        )
        robot.apply_action(action)

    def step(self, render: bool = True) -> None:
        """Steps the physics simulation while rendering or without.
        Args:
            render (bool, optional): Set to False to only do a physics simulation without rendering. Note:
                                     app UI will be frozen (since its not rendering) in this case.
                                     Defaults to True.

        Raises:
            Exception: if there is no stage currently opened

        """
        self.world.step(render=render)

    def close(self) -> None:
        """Closes the simulator."""
        self.simulator.close()


def main():
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config/default_config.toml")
    args = parser.parse_args()
    # Initialize the sim core
    sim_core = SimCore(config=args.config)

    # Run sim core test
    sim_core.run_display()


if __name__ == "__main__":
    main()
