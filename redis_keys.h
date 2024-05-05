#pragma once

// - read:
const std::string JOINT_ANGLES_WALLE_KEY = "cs225a::robot_q::Panda1";
const std::string JOINT_VELOCITIES_WALLE_KEY = "cs225a::robot_dq::Panda2";
const std::string JOINT_ANGLES_EVE_KEY = "cs225a::robot_q::Panda1";
const std::string JOINT_VELOCITIES_EVE_KEY = "cs225a::robot_dq::Panda2";
// - write
const std::string JOINT_TORQUES_COMMANDED_WALLE_KEY = "cs225a::robot_command_torques::Panda1";
const std::string JOINT_TORQUES_COMMANDED_EVE_KEY = "cs225a::robot_command_torques::Panda2";
const std::string GRAVITY_COMP_ENABLED_KEY = "cs225a::simviz::gravity_comp_enabled";
