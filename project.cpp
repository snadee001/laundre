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

// sai2 utilities from sai2-common
#include "timer/LoopTimer.h"
#include "redis/RedisClient.h"

// redis keys
#include "redis_keys.h"

// States 
enum State {
    LEFT_REACH,
    LEFT_GRASP,
    LEFT_RAISE,
    LEFT_TOCENTER,
    RIGHT_REACH,
    RIGHT_GRASP,
    RIGHT_RAISE,
    RIGHT_TOCENTER
};

// for handling ctrl+c and interruptions properly
#include <signal.h>
bool runloop = true;
void sighandler(int) { runloop = false; }

// namespaces for compactness of code
using namespace std;
using namespace Eigen;
using namespace Sai2Primitives;
using namespace ruckig;

// config file names and object names
const string robot_file = "${CS225A_URDF_FOLDER}/panda/panda_arm_hand.urdf";

Vector3d move(Vector3d x_desired, Vector3d spacing) {
    Vector3d walle_origin = Vector3d(0, -0.3, 0);

    return x_desired - walle_origin - spacing;
}

Vector2d grasp(bool is_grasp) {
    if (is_grasp) return Vector2d(0.02, -0.02);
    else return Vector2d(0.05, -0.05);
}


int main(int argc, char** argv) {

    int state = LEFT_REACH;
    bool start = true;

    Sai2Model::URDF_FOLDERS["CS225A_URDF_FOLDER"] = string(CS225A_URDF_FOLDER);

    // set up signal handler
    signal(SIGABRT, &sighandler);
    signal(SIGTERM, &sighandler);
    signal(SIGINT, &sighandler);

    // load robots
    auto walle = std::make_shared<Sai2Model::Sai2Model>(robot_file);
    auto eve = std::make_shared<Sai2Model::Sai2Model>(robot_file);

    std::string custom_box_file = "${CS225A_URDF_FOLDER}/custom_box/custom_box.urdf";
    auto custom_box = std::make_shared<Sai2Model::Sai2Model>(custom_box_file);

    Vector3d eve_origin = Vector3d(0, 0.3, 0);
    Vector3d walle_origin = Vector3d(0, -0.3, 0);

    // prepare controller
	int dof = walle->dof(); // same as eve here
	VectorXd walle_command_torques = VectorXd::Zero(dof);  // panda + gripper torques 
	MatrixXd walle_N_prec = MatrixXd::Identity(dof, dof);

    int custom_box_dof = custom_box->dof();
    VectorXd custom_box_command_torques = VectorXd::Zero(custom_box_dof);
    MatrixXd custom_box_N_prec = MatrixXd::Identity(custom_box_dof, custom_box_dof);

    // arm task (joints 0-6)
    const string control_link = "link7";
	const Vector3d control_point = Vector3d(0, 0, 0.0);
	Affine3d compliant_frame = Affine3d::Identity();
	compliant_frame.translation() = control_point;

	auto walle_pose_task = std::make_shared<Sai2Primitives::MotionForceTask>(walle, control_link, compliant_frame);
	walle_pose_task->setPosControlGains(400, 40, 0);
	walle_pose_task->setOriControlGains(400, 40, 0);

    Vector3d walle_ee_pos;
	Matrix3d walle_ee_ori;

    // gripper partial joint task (joints 7-8)
    MatrixXd gripper_selection_matrix = MatrixXd::Zero(2, walle->dof());
	gripper_selection_matrix(0, 7) = 1;
	gripper_selection_matrix(1, 8) = 1;

	double kp_gripper = 5e3;
	double kv_gripper = 1e2;

	auto walle_gripper_task = std::make_shared<Sai2Primitives::JointTask>(walle, gripper_selection_matrix);
	walle_gripper_task->setDynamicDecouplingType(Sai2Primitives::DynamicDecouplingType::IMPEDANCE);
	walle_gripper_task->setGains(kp_gripper, kv_gripper, 0);

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

    auto custom_box_joint_task = std::make_shared<Sai2Primitives::JointTask>(custom_box);
    custom_box_joint_task->setGains(kp_j, kv_j, 0);
    VectorXd custom_box_q_desired = VectorXd::Zero(custom_box_dof);
    custom_box_joint_task->setGoalPosition(custom_box_q_desired);

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

    VectorXd box_q = redis_client.getEigen(BOX_ANGLES_KEY);
    redis_client.addToReceiveGroup(BOX_ANGLES_KEY, box_q);

    //MatrixXd box_pose = redis_client.getEigen(BOX_POSE_KEY);
    //redis_client.addToReceiveGroup(BOX_POSE_KEY, box_pose);

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

    ofstream file;
    file.open("../../homework/laundre/data_files/logger.txt");\

    //Variables for folding task
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

        Matrix3d walle_R_desired;
        walle_R_desired << 1.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, -1.0;

        Vector3d walle_x_desired = walle_ee_pos;

        Vector2d gripper_desired;
        Matrix3d ee_ori_world;
        Matrix3d box_ori = Matrix3d::Zero();

        if (start) {
            if (state == LEFT_REACH) {
                Vector3d x_desired = custom_box->positionInWorld("flap_left")+Vector3d(0.1, -0.1, 0.15);
                walle_x_desired = move(x_desired, Vector3d(0.0, 0, 0));
                gripper_desired = grasp(false);
            }
            else if (state == LEFT_GRASP) {
                Vector3d x_desired = custom_box->positionInWorld("flap_left")+Vector3d(0.1, -0.2, 0.15);
                walle_x_desired = move(x_desired, Vector3d(0.0, 0, 0));
                gripper_desired = grasp(true);
            }
            else if (state == LEFT_RAISE) {
                Vector3d x_desired = custom_box->positionInWorld("box_base")+Vector3d(0, -0.1, 0.3);
                walle_x_desired = move(x_desired, Vector3d(0.0, 0, 0));
                gripper_desired = grasp(true);
            }
            else if (state == LEFT_TOCENTER) {
                Vector3d x_desired = custom_box->positionInWorld("box_base")+Vector3d(0, 0, 0.2);
                walle_x_desired = move(x_desired, Vector3d(0.0, 0, 0));
                gripper_desired = grasp(true);
            }
            else if (state == RIGHT_REACH) {
                Vector3d x_desired = custom_box->positionInWorld("flap_right")+Vector3d(0.1, 0.1, 0.15);
                walle_x_desired = move(x_desired, Vector3d(0.0, 0, 0));
                gripper_desired = grasp(false);
            }
            else if (state == RIGHT_GRASP) {
                Vector3d x_desired = custom_box->positionInWorld("flap_right")+Vector3d(0.1, 0.2, 0.15);
                walle_x_desired = move(x_desired, Vector3d(0.0, 0, 0));
                gripper_desired = grasp(true);
            }
            else if (state == RIGHT_RAISE) {
                Vector3d x_desired = custom_box->positionInWorld("box_base")+Vector3d(0, 0.1, 0.3);
                walle_x_desired = move(x_desired, Vector3d(0.0, 0, 0));
                gripper_desired = grasp(true);
            }
            else if (state == RIGHT_TOCENTER) {
                Vector3d x_desired = custom_box->positionInWorld("box_base")+Vector3d(0, 0, 0.2);
                walle_x_desired = move(x_desired, Vector3d(0.0, 0, 0));
                gripper_desired = grasp(true);
            }
            else {
                gripper_desired = grasp(false);
                cout << "Done!" << "\n";
            }

            walle_pose_task->setGoalPosition(walle_x_desired);
            walle_gripper_task->setGoalPosition(gripper_desired);

            if (state != LEFT_REACH && state != RIGHT_REACH) {
                walle_pose_task->setGoalOrientation(walle_R_desired);
            }

            start = false;
            if (state <= RIGHT_TOCENTER)
                cout << "Started state " << state << "\n";
        }
        else if (walle_pose_task->goalPositionReached(0.01) 
        && walle_pose_task->goalOrientationReached(0.01)
        && walle_gripper_task->goalPositionReached()
        && state <= RIGHT_TOCENTER) {
            cout << "Moving on to next state! Yay!" << "\n";
            state += 1;
            start = true;
        }
        

        // update task model
        walle_N_prec.setIdentity();
        walle_pose_task->updateTaskModel(walle_N_prec);
        walle_gripper_task->updateTaskModel(walle_pose_task->getTaskAndPreviousNullspace());
        walle_joint_task->updateTaskModel(walle_gripper_task->getTaskAndPreviousNullspace());

        // compute torques
        walle_command_torques = walle_pose_task->computeTorques() + walle_gripper_task->computeTorques() + walle_joint_task->computeTorques();


        // send to redis
        redis_client.sendAllFromGroup();
    }

    if (file.is_open()) {
        file.close();
    }

    walle_command_torques.setZero();
    gravity_comp_enabled = true;
    redis_client.sendAllFromGroup();

    timer.stop();
    cout << "\nControl loop timer stats:\n";
    timer.printInfoPostRun();

    return 0;
}
