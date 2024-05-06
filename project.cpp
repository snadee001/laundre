// some standard library includes
#include <math.h>

#include <fstream>
#include <iostream>
#include <mutex>
#include <string>
#include <thread>

// sai2 main libraries includes
#include "Sai2Model.h"

// sai2 utilities from sai2-common
#include "timer/LoopTimer.h"
#include "redis/RedisClient.h"

// redis keys
#include "redis_keys.h"

// for handling ctrl+c and interruptions properly
#include <signal.h>
bool runloop = true;
void sighandler(int) { runloop = false; }

// namespaces for compactness of code
using namespace std;
using namespace Eigen;

// config file names and object names
const string robot_file = "${CS225A_URDF_FOLDER}/panda/panda_arm.urdf";

int main(int argc, char** argv) {
    Sai2Model::URDF_FOLDERS["CS225A_URDF_FOLDER"] = string(CS225A_URDF_FOLDER);

    // set up signal handler
    signal(SIGABRT, &sighandler);
    signal(SIGTERM, &sighandler);
    signal(SIGINT, &sighandler);

    // load robots
    auto walle = new Sai2Model::Sai2Model(robot_file);
    auto eve = new Sai2Model::Sai2Model(robot_file);

    Vector3d eve_origin = Vector3d(0, 0.3, 0);
    Vector3d walle_origin = Vector3d(0, -0.3, 0);

    // prepare controller
	int dof = walle->dof();
	const string link_name = "link7";
	const Vector3d pos_in_link = Vector3d(0, 0, 0.15);
	VectorXd walle_control_torques = VectorXd::Zero(dof);
    VectorXd eve_control_torques = VectorXd::Zero(dof);

	// model quantities for operational space control
	MatrixXd walle_Jv = MatrixXd::Zero(3,dof);
	MatrixXd walle_Lambda = MatrixXd::Zero(3,3);
	MatrixXd walle_J_bar = MatrixXd::Zero(dof,3);
	MatrixXd walle_N = MatrixXd::Zero(dof,dof);

    MatrixXd eve_Jv = MatrixXd::Zero(3,dof);
	MatrixXd eve_Lambda = MatrixXd::Zero(3,3);
	MatrixXd eve_J_bar = MatrixXd::Zero(dof,3);
	MatrixXd eve_N = MatrixXd::Zero(dof,dof);

	walle_Jv = walle->Jv(link_name, pos_in_link);
	walle_Lambda = walle->taskInertiaMatrix(walle_Jv);
	walle_J_bar = walle->dynConsistentInverseJacobian(walle_Jv);
	walle_N = walle->nullspaceMatrix(walle_Jv);

    eve_Jv = eve->Jv(link_name, pos_in_link);
	eve_Lambda = eve->taskInertiaMatrix(eve_Jv);
	eve_J_bar = eve->dynConsistentInverseJacobian(eve_Jv);
	eve_N = eve->nullspaceMatrix(eve_Jv);

    // flag for enabling gravity compensation
    bool gravity_comp_enabled = false;

    // start redis client
    auto redis_client = Sai2Common::RedisClient();
    redis_client.connect();

    // setup send and receive groups
    VectorXd walle_q = redis_client.getEigen(JOINT_ANGLES_WALLE_KEY);
    VectorXd walle_dq = redis_client.getEigen(JOINT_VELOCITIES_WALLE_KEY);
    redis_client.addToReceiveGroup(JOINT_ANGLES_WALLE_KEY, walle_q);
    redis_client.addToReceiveGroup(JOINT_VELOCITIES_WALLE_KEY, walle_dq);

    VectorXd eve_q = redis_client.getEigen(JOINT_ANGLES_EVE_KEY);
    VectorXd eve_dq = redis_client.getEigen(JOINT_VELOCITIES_EVE_KEY);
    redis_client.addToReceiveGroup(JOINT_ANGLES_EVE_KEY, eve_q);
    redis_client.addToReceiveGroup(JOINT_VELOCITIES_EVE_KEY, eve_dq);

    MatrixXd box_pose = redis_client.getEigen(BOX_POSE_KEY);
    redis_client.addToReceiveGroup(BOX_POSE_KEY, box_pose);

    redis_client.addToSendGroup(JOINT_TORQUES_COMMANDED_WALLE_KEY, walle_control_torques);
    redis_client.addToSendGroup(JOINT_TORQUES_COMMANDED_EVE_KEY, eve_control_torques);
    redis_client.addToSendGroup(GRAVITY_COMP_ENABLED_KEY, gravity_comp_enabled);

    redis_client.receiveAllFromGroup();
    redis_client.sendAllFromGroup();

    // update robot model from simulation configuration
    walle->setQ(walle_q);
    walle->setDq(walle_dq);
    walle->updateModel();

    eve->setQ(eve_q);
    eve->setDq(eve_dq);
    eve->updateModel();

    // record initial configuration
    VectorXd initial_walle_q = walle->q();
    VectorXd initial_eve_q = eve->q();

    // create a loop timer
    const double control_freq = 1000;
    Sai2Common::LoopTimer timer(control_freq);

    ofstream file;
    file.open("../../homework/laundre/data_files/logger.txt");

    while (runloop) {
        // wait for next scheduled loop
        timer.waitForNextLoop();
        double time = timer.elapsedTime();

        // read robot state from redis
        redis_client.receiveAllFromGroup();
        walle->setQ(walle_q);
        walle->setDq(walle_dq);
        walle->updateModel();

        eve->setQ(eve_q);
        eve->setDq(eve_dq);
        eve->updateModel();

        // **********************
        // WRITE YOUR CODE AFTER
        // **********************

        walle_Jv = walle->Jv(link_name, pos_in_link);
        walle_Lambda = walle->taskInertiaMatrix(walle_Jv);
        walle_J_bar = walle->dynConsistentInverseJacobian(walle_Jv);
        walle_N = walle->nullspaceMatrix(walle_Jv);

        eve_Jv = eve->Jv(link_name, pos_in_link);
        eve_Lambda = eve->taskInertiaMatrix(eve_Jv);
        eve_J_bar = eve->dynConsistentInverseJacobian(eve_Jv);
        eve_N = eve->nullspaceMatrix(eve_Jv);
        
        // **********************
        // CONTROL WALLE & EVE CODE HERE
        // **********************

        double kp = 10.0;      // chose your p gain
        double kv = 5.0;      // chose your d gain
        double kvj = 2.0;

        Vector3d xd = box_pose(seq(0,2), 3);

        Vector3d eve_x = eve->position(link_name, pos_in_link)+eve_origin;
        Vector3d eve_v = eve_Jv*eve_dq;
        Vector3d walle_x = walle->position(link_name, pos_in_link)+walle_origin;
        Vector3d walle_v = walle_Jv*walle_dq;

        Vector3d eve_F = eve->taskInertiaMatrix(eve_Jv)*(-1*kp*(eve_x-xd)-kv*eve_v);
        eve_control_torques = eve_Jv.transpose()*eve_F+eve->jointGravityVector()-eve->nullspaceMatrix(eve_Jv).transpose()*eve->M()*(kvj*eve_dq);

        Vector3d walle_F = walle->taskInertiaMatrix(walle_Jv)*(-1*kp*(walle_x-xd)-kv*walle_v);
        walle_control_torques = walle_Jv.transpose()*walle_F+walle->jointGravityVector()-walle->nullspaceMatrix(walle_Jv).transpose()*walle->M()*(kvj*walle_dq);

        // send to redis
        redis_client.sendAllFromGroup();
    }

    if (file.is_open()) {
        file.close();
    }

    walle_control_torques.setZero();
    eve_control_torques.setZero();
    gravity_comp_enabled = true;
    redis_client.sendAllFromGroup();

    timer.stop();
    cout << "\nControl loop timer stats:\n";
    timer.printInfoPostRun();

    return 0;
}
