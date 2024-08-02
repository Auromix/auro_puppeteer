from auro_puppeteer.human_hand_processor.human_hand_processor import HumanHandProcessor
from auro_puppeteer.simulators.isaac_sim.core import SimCore
from scipy.spatial.transform import Rotation
import numpy as np

import cv2


def main():
    config = "config/rgb_camera_dual_hand_teleoperation.toml"
    cap_path = 0

    # Init camera
    if cap_path is None:
        cap_path = 0
    cap = cv2.VideoCapture(cap_path)
    # Init hand processor
    human_hand_processor = HumanHandProcessor(config=config)
    # Init sim core
    sim_core = SimCore(config=config)
    # Get hand robot
    right_hand = sim_core.robot_group["right_hand"]
    left_hand = sim_core.robot_group["left_hand"]

    # Get joint index
    left_hand_joint_indexes = (
        human_hand_processor.get_joint_indexes_for_dex_retargeting_in_isaac_sim(
            hand="left_hand", robot=left_hand
        )
    )
    right_hand_joint_indexes = (
        human_hand_processor.get_joint_indexes_for_dex_retargeting_in_isaac_sim(
            hand="right_hand", robot=right_hand
        )
    )
    # left_hand_joint_names = (
    #     human_hand_processor.processor.left_hand_dex_retargeting.joint_names
    # )
    # right_hand_joint_names = (
    #     human_hand_processor.processor.right_hand_dex_retargeting.joint_names
    # )
    # sim_left_hand_joint_names = [
    #     left_hand.dof_names[index] for index in left_hand_joint_indexes
    # ]
    # sim_right_hand_joint_names = [
    #     right_hand.dof_names[index] for index in right_hand_joint_indexes
    # ]
    # # Log left
    # sim_core.logger.log_info(f"Dex retargeting left hand joint names: {left_hand_joint_names}")
    # sim_core.logger.log_info(f"Dex retargeting left hand joint indexes: {left_hand_joint_indexes}")
    # sim_core.logger.log_info(f"Sim left hand joint names:{sim_left_hand_joint_names}")
    # sim_core.logger.log_info(f"Sim left hand joint indexes: {[left_hand.get_dof_index(name) for name in left_hand.dof_names]}")

    # # Log right
    # sim_core.logger.log_info(f"Dex retargeting right hand joint names: {right_hand_joint_names}")
    # sim_core.logger.log_info(f"Dex retargeting right hand joint indexes: {right_hand_joint_indexes}")
    # sim_core.logger.log_info(f"Sim right hand joint names:{sim_right_hand_joint_names}")
    # sim_core.logger.log_info(f"Sim right hand joint indexes: {[right_hand.get_dof_index(name) for name in right_hand.dof_names]}")

    while cap.isOpened() and sim_core.simulator.is_running():
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

        # Control robot
        if right_hand_joint_angles is not None:
            # Set wrist rotation
            # x-backward, y-upward to z-forward, x-downward
            # detection_frame_to_hand_base_frame_rotation=Rotation.from_euler('XYZ',[0,0,0],degrees=True)
            # right_wrist_rotation = np.dot(detection_frame_to_hand_base_frame_rotation.as_matrix(),right_wrist_rotation)

            # orientation = Rotation.from_matrix(right_wrist_rotation).as_quat()
            # orientation= orientation[[3,0,1,2]]

            # right_hand.set_world_pose(position=None,orientation=orientation)

            # Set hand joint positions
            right_hand.set_joint_positions(
                positions=right_hand_joint_angles,
                joint_indices=right_hand_joint_indexes,
            )

            # sim_core.set_robot_joint_angles(joint_angles=right_hand_joint_angles, joint_indexes=right_hand_joint_indexes,robot=right_hand)

        if left_hand_joint_angles is not None:
            # Set wrist rotation
            # x-backward, y-upward to z-forward, x-downward
            # detection_frame_to_hand_base_frame_rotation=Rotation.from_euler('XYZ',[0,0,0],degrees=True)
            # left_wrist_rotation = np.dot(detection_frame_to_hand_base_frame_rotation.as_matrix(),left_wrist_rotation)

            # orientation = Rotation.from_matrix(left_wrist_rotation).as_quat()
            # orientation= orientation[[3,0,1,2]]

            # left_hand.set_world_pose(position=None,orientation=orientation)

            # Set hand joint positions
            left_hand.set_joint_positions(
                positions=left_hand_joint_angles, joint_indices=left_hand_joint_indexes
            )

            # sim_core.set_robot_joint_angles(joint_angles=left_hand_joint_angles,joint_indexes=left_hand_joint_indexes,robot=left_hand)

        # Step
        sim_core.step()

        cv2.imshow("frame", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
