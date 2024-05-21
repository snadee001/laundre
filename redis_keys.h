#pragma once

// - read:
const std::string JOINT_ANGLES_WALLE_KEY = "cs225a::robot_q::WALLE";
const std::string JOINT_VELOCITIES_WALLE_KEY = "cs225a::robot_dq::WALLE";
const std::string JOINT_ANGLES_EVE_KEY = "cs225a::robot_q::EVE";
const std::string JOINT_VELOCITIES_EVE_KEY = "cs225a::robot_dq::EVE";
const std::string BOX_ANGLES_KEY = "cs225a::robot_q::custom_box";

// - write
const std::string JOINT_TORQUES_COMMANDED_WALLE_KEY = "cs225a::robot_command_torques::WALLE";
const std::string JOINT_TORQUES_COMMANDED_EVE_KEY = "cs225a::robot_command_torques::EVE";
const std::string GRAVITY_COMP_ENABLED_KEY = "cs225a::simviz::gravity_comp_enabled";
