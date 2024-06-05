#pragma once

// - read:
std::string JOINT_ANGLES_WALLE_KEY = "cs225a::robot_q::WALLE";
std::string JOINT_VELOCITIES_WALLE_KEY = "cs225a::robot_dq::WALLE";
std::string JOINT_ANGLES_EVE_KEY = "cs225a::robot_q::EVE";
std::string JOINT_VELOCITIES_EVE_KEY = "cs225a::robot_dq::EVE";
std::string WALLE_EE_FORCE = "cs225a::force_sensor::WALLE::end-effector::force";
std::string WALLE_EE_MOMENT = "cs225a::force_sensor::WALLE::end-effector::moment";

// - write
std::string JOINT_TORQUES_COMMANDED_WALLE_KEY = "cs225a::robot_command_torques::WALLE";
std::string JOINT_TORQUES_COMMANDED_EVE_KEY = "cs225a::robot_command_torques::EVE";
std::string GRAVITY_COMP_ENABLED_KEY = "cs225a::simviz::gravity_comp_enabled";
