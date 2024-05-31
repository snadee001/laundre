// some standard library includes
#include <math.h>
#include <fstream>
#include <iostream>
#include <mutex>
#include <string>
#include <thread>
#include <tuple>

// sai2 main libraries includes
#include "Sai2Model.h"
#include "Sai2Primitives.h"
#include "Sai2Simulation.h"
#include "force_sensor_sim/ForceSensorSim.h"

// sai2 utilities from sai2-common
#include "timer/LoopTimer.h"
#include "redis/RedisClient.h"

// redis keys
#include "redis_keys.h"

// States
enum Part {
    LEFT_SLEEVE,
    RIGHT_SLEEVE,
    BOTTOM_LEFT,
    BOTTOM_RIGHT,
    BOTTOM,
    BOTTOM2,
    THIRD,
    THIRD2
};

enum Action {
    REACH,
    SLIP,
    FLIP,
    RISE
};

// for handling ctrl+c and interruptions properly
#include <signal.h>
bool runloop = true;
void sighandler(int) { runloop = false; }

// namespaces for compactness of code
using namespace std;
using namespace Eigen;
using namespace Sai2Primitives;

// config file names and object names
const string robot_file = "${HW_FOLDER}/laundre/panda/panda_arm_spatula.urdf";

Matrix3d ori(Vector3d cur, Vector3d target) {
    Matrix3d result = Matrix3d::Zero();
    result(2, 2) = -1.0;
    Vector2d vec = (target - cur).normalized().head(2);
    result(0, 0) = vec(0);
    result(1, 0) = vec(1);
    result(0, 1) = vec(1);
    result(1, 1) = -vec(0);
    return result;
}

int main(int argc, char** argv) {

    int part = LEFT_SLEEVE;
    int action = REACH;
    bool start = true;

    Sai2Model::URDF_FOLDERS["CS225A_URDF_FOLDER"] = string(CS225A_URDF_FOLDER);
    Sai2Model::URDF_FOLDERS["HW_FOLDER"] = string(HW_FOLDER);

    // set up signal handler
    signal(SIGABRT, &sighandler);
    signal(SIGTERM, &sighandler);
    signal(SIGINT, &sighandler);

    // load robots
    auto walle = std::make_shared<Sai2Model::Sai2Model>(robot_file);

    // prepare controller
    int dof = walle->dof(); // same as eve here
    VectorXd walle_command_torques = VectorXd::Zero(dof);  // panda + gripper torques 
    MatrixXd walle_N_prec = MatrixXd::Identity(dof, dof);

    // arm task (joints 0-6)
    const string control_link = "end-effector";
    const Vector3d control_point = Vector3d(0.2286, 0.0, 0.01);
    Affine3d compliant_frame = Affine3d::Identity();
    compliant_frame.translation() = control_point;

    auto walle_pose_task = std::make_shared<Sai2Primitives::MotionForceTask>(walle, control_link, compliant_frame);
    walle_pose_task->setPosControlGains(400, 40, 0);
    walle_pose_task->setOriControlGains(400, 40, 0);

    Vector3d walle_ee_pos;
    Matrix3d walle_ee_ori;

    // joint task for posture control
    auto walle_joint_task = std::make_shared<Sai2Primitives::JointTask>(walle);

    double kp_j = 400.0;
    double kv_j = 40.0;
    walle_joint_task->setGains(kp_j, kv_j, 0);

    VectorXd q_desired(dof);
    q_desired.head(7) << -30.0, -15.0, -15.0, -105.0, 0.0, 90.0, 45.0;
    q_desired.head(7) *= M_PI / 180.0;
    q_desired.tail(2) << 0.04, -0.04;
    walle_joint_task->setGoalPosition(q_desired);

    // flag for enabling gravity compensation
    bool gravity_comp_enabled = true;

    // start redis client
    auto redis_client = Sai2Common::RedisClient();
    redis_client.connect();

    // setup send and receive groups
    VectorXd walle_q = redis_client.getEigen(JOINT_ANGLES_WALLE_KEY);
    VectorXd walle_dq = redis_client.getEigen(JOINT_VELOCITIES_WALLE_KEY);
    redis_client.addToReceiveGroup(JOINT_ANGLES_WALLE_KEY, walle_q);
    redis_client.addToReceiveGroup(JOINT_VELOCITIES_WALLE_KEY, walle_dq);

    redis_client.addToSendGroup(JOINT_TORQUES_COMMANDED_WALLE_KEY, walle_command_torques);
    redis_client.addToSendGroup(GRAVITY_COMP_ENABLED_KEY, gravity_comp_enabled);

    redis_client.receiveAllFromGroup();
    redis_client.sendAllFromGroup();

    // update robot model from simulation configuration
    walle->setQ(walle_q);
    walle->setDq(walle_dq);
    walle->updateModel();

    // record initial configuration
    VectorXd initial_walle_q = walle->q();

    // create a loop timer
    const double control_freq = 1000;
    Sai2Common::LoopTimer timer(control_freq);

    // Variables for folding task
    const double control_cycle {0.001};
    Vector3d offset;
    Ruckig<3, EigenVector> otg {control_cycle};
    InputParameter<3, EigenVector> input;
    OutputParameter<3, EigenVector> output;

    while (runloop) {
        // wait for next scheduled loop
        timer.waitForNextLoop();
        double time = timer.elapsedTime();

        // read robot state from redis
        redis_client.receiveAllFromGroup();
        walle->setQ(walle_q);
        walle->setDq(walle_dq);
        walle->updateModel();

        walle_ee_pos = walle->position(control_link, control_point);
        walle_ee_ori = walle->rotation(control_link);

        Matrix3d walle_R_desired = walle_ee_ori;
        Matrix3d rotation135;
        rotation135 <<
            sqrt(2.0)/-2.0, 0.0, sqrt(2.0)/-2.0,
            0.0, 1.0, 0.0,
            sqrt(2.0)/2.0, 0, sqrt(2.0)/-2.0;

        Vector3d walle_x_desired = walle_ee_pos;

        Matrix3d ee_ori_world;
        Matrix3d box_ori = Matrix3d::Zero();
        Vector3d x_cur, x_target;

        
        if (start) {
            // x_cur and x_target should be hooked up to CV pipeline & determined
            if (part == LEFT_SLEEVE) {
                x_cur = Vector3d(0.5, -0.2, 0.0);
                x_target = Vector3d(0.5, -0.1, 0.0);
            } else if (part == RIGHT_SLEEVE) {
                x_cur = Vector3d(0.5, 0.2, 0.0);
                x_target = Vector3d(0.5, 0.1, 0.0);
            } else if (part == BOTTOM_LEFT) {
                x_cur = Vector3d(0.5, 0.2, 0.0);
                x_target = Vector3d(0.5, 0.1, 0.0);
            } else if (part == BOTTOM_RIGHT) {
                x_cur = Vector3d(0.5, 0.2, 0.0);
                x_target = Vector3d(0.5, 0.1, 0.0);
            } else if (part == BOTTOM) {
                x_cur = Vector3d(0.5, 0.2, 0.0);
                x_target = Vector3d(0.5, 0.1, 0.0);
            } else if (part == BOTTOM2) {
                x_cur = Vector3d(0.5, 0.2, 0.0);
                x_target = Vector3d(0.5, 0.1, 0.0);
            } else if (part == THIRD) {
                x_cur = Vector3d(0.5, 0.2, 0.0);
                x_target = Vector3d(0.5, 0.1, 0.0);
            } else if (part == THIRD2) {
                x_cur = Vector3d(0.5, 0.2, 0.0);
                x_target = Vector3d(0.5, 0.1, 0.0);
            } else {
                x_cur = Vector3d(0.0, 0.0, 0.2); // move above workspace when done
                x_target = Vector3d(0.1, 0.0, 0.2);
            }

            if (action == REACH) {
                walle_pose_task->setGoalPosition(x_cur);
                walle_pose_task->setGoalOrientation(ori(x_cur, x_target));
            } else if (action == SLIP) {
                walle_pose_task->parametrizeForceMotionSpaces(
					1, Vector3d::UnitZ());

				// set the force control 
				walle_pose_task->setGoalForce(-1.0 * Vector3d::UnitZ());

				walle_pose_task->setForceControlGains(0.7, 5.0, 1.5);

                walle_pose_task->setGoalPosition(x_target);
                walle_pose_task->setGoalOrientation(ori(x_cur, x_target));
            } else if (action == FLIP) {
                // Keep end effector position fixed, only change orientation
                walle_pose_task->setGoalPosition(x_target);
                walle_pose_task->setGoalOrientation(walle_ee_ori * rotation135 * walle_ee_ori.inverse() * ori(x_cur, x_target));
            } else if (action == RISE) {
                walle_pose_task->parametrizeForceMotionSpaces(0);
                walle_pose_task->setGoalPosition(x_target + Vector3d(0, 0, 0.3));
            }

            start = false;
            if (part <= THIRD2) {
                cout << "Working on action " << action << " with part " << part << "\n";
            } else {
                cout << "Done!" << "\n";
            }
        } else if (walle_pose_task->goalPositionReached(0.01) && walle_pose_task->goalOrientationReached(0.01) && part <= THIRD2) {
            cout << "Moving on! Yay!" << "\n";
            action = (action + 1) % 4;
            if (action == 0) part += 1;
            start = true;
        }

        // update task model
        walle_N_prec.setIdentity();
        walle_pose_task->updateTaskModel(walle_N_prec);
        walle_joint_task->updateTaskModel(walle_pose_task->getTaskAndPreviousNullspace());

        // compute torques
        walle_command_torques = walle_joint_task->computeTorques() + walle_pose_task->computeTorques();

        // send to redis
        redis_client.sendAllFromGroup();
    }

    walle_command_torques.setZero();
    gravity_comp_enabled = true;
    redis_client.sendAllFromGroup();

    timer.stop();
    cout << "\nControl loop timer stats:\n";
    timer.printInfoPostRun();

    return 0;
}
