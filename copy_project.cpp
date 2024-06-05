// some standard library includes
#include <math.h>
#include <fstream>
#include <iostream>
#include <mutex>
#include <string>
#include <thread>
#include <tuple>
#include <cstdlib>

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
const bool flag_simulation = false;

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
    // set up signal handler
    signal(SIGABRT, &sighandler);
    signal(SIGTERM, &sighandler);
    signal(SIGINT, &sighandler); 


    float camera_height = 0.08;
    //int vision = 0;
    //while (vision == 0) vision = system("python3 vision.py");
    ifstream points_file("points.txt"); //Calibrated xy coordinates in robot base frame
    string num;
    int points[8][2][2]; //Part / cur-target / x-y
    int i=0, j=0, k=0;
    if (points_file.is_open()) while(points_file) {
        points_file >> num;
        points[i][j][k] = stoi(num);

        k = 1-k;
        if (k==0) {
            j = 1-j;
            if (j==0) i += 1;
        }
    }

    int part = LEFT_SLEEVE;
    int action = REACH;
    bool start = true;

    Sai2Model::URDF_FOLDERS["CS225A_URDF_FOLDER"] = string(CS225A_URDF_FOLDER);
    Sai2Model::URDF_FOLDERS["HW_FOLDER"] = string(HW_FOLDER);

    // load robots
    auto walle = std::make_shared<Sai2Model::Sai2Model>(robot_file);

    // prepare controller
    int dof = walle->dof(); // same as eve here
    VectorXd walle_command_torques = VectorXd::Zero(dof);  // panda + gripper torques 
    MatrixXd walle_N_prec = MatrixXd::Identity(dof, dof);

    // arm task (joints 0-6)
    const string control_link = "end-effector";
    const Vector3d control_point = Vector3d(0.195, -0.0, 0.0785);
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

    // set up signal handler
    signal(SIGABRT, &sighandler);
    signal(SIGTERM, &sighandler);
    signal(SIGINT, &sighandler);

    std::string MASSMATRIX_KEY = "sai2::FrankaPanda::Clyde::sensors::model::massmatrix"; 

	// keys 
	if (!flag_simulation) {
		JOINT_TORQUES_COMMANDED_WALLE_KEY = "sai2::FrankaPanda::Clyde::actuators::fgc";
		JOINT_ANGLES_WALLE_KEY = "sai2::FrankaPanda::Clyde::sensors::q";
		JOINT_VELOCITIES_WALLE_KEY = "sai2::FrankaPanda::Clyde::sensors::dq";
		//MASSMATRIX_KEY = "sai2::FrankaPanda::Clyde::sensors::model::massmatrix";  // NEED TO MAKE THIS
		//GRAVITY_COMP_ENABLED_KEY = "sai2::FrankaPanda::Clyde::sensors::model::robot_gravity";
	}

    // setup send and receive groups
    VectorXd walle_q = redis_client.getEigen(JOINT_ANGLES_WALLE_KEY);
    VectorXd walle_dq = redis_client.getEigen(JOINT_VELOCITIES_WALLE_KEY);
    MatrixXd mass_matrix = redis_client.getEigen(MASSMATRIX_KEY);
    redis_client.addToReceiveGroup(JOINT_ANGLES_WALLE_KEY, walle_q);
    redis_client.addToReceiveGroup(JOINT_VELOCITIES_WALLE_KEY, walle_dq);
    redis_client.addToReceiveGroup(MASSMATRIX_KEY, mass_matrix);

    redis_client.addToSendGroup(JOINT_TORQUES_COMMANDED_WALLE_KEY, walle_command_torques);
    //redis_client.addToSendGroup(GRAVITY_COMP_ENABLED_KEY, gravity_comp_enabled);

    redis_client.receiveAllFromGroup();
    redis_client.sendAllFromGroup();

    // update robot model from simulation configuration
    walle->setQ(walle_q);
    walle->setDq(walle_dq);

    if (!flag_simulation) {
        cout << "creating mass matrix" << endl;
        MatrixXd M = redis_client.getEigen(MASSMATRIX_KEY);
        M(4, 4) += 0.2;
        M(5, 5) += 0.2;
        M(6, 6) += 0.2;
        walle->updateModel(M);
    }

    else {
        walle->updateModel();
    }

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
        double time = timer.elapsedSimTime();

        // read robot state from redis
        redis_client.receiveAllFromGroup();

        if (!flag_simulation) {
            cout << "here" << endl;
            walle_q.head(7) = redis_client.getEigen(JOINT_ANGLES_WALLE_KEY);
            walle_dq.head(7) = redis_client.getEigen(JOINT_VELOCITIES_WALLE_KEY);
        }
        else {
            walle_q = redis_client.getEigen(JOINT_ANGLES_WALLE_KEY);
            walle_dq = redis_client.getEigen(JOINT_VELOCITIES_WALLE_KEY);
        }

        walle->setQ(walle_q);
        walle->setDq(walle_dq);

        if (!flag_simulation) {
            cout << "creating mass matrix" << endl;
            MatrixXd M = redis_client.getEigen(MASSMATRIX_KEY);
            M(4, 4) += 0.2;
            M(5, 5) += 0.2;
            M(6, 6) += 0.2;
            walle->updateModel(M);

        } else {
            cout << "updates model in else statement" << endl;
            walle->updateModel();
        }

        walle_ee_pos = walle->position(control_link, control_point);
        walle_ee_ori = walle->rotation(control_link);
        Matrix3d walle_R_desired = walle_ee_ori;
        Vector3d walle_x_desired = walle_ee_pos;


        Matrix3d ee_ori_world;
        Matrix3d box_ori = Matrix3d::Zero();
        Vector3d x_cur, x_target;

        cout << "start's value: " << start << endl;
        cout << "reach's value: " << REACH << endl;


        cout << "entered start" << endl;
        x_cur =  Vector3d(0.389199, -0.213982, 0.053225);
        x_target = Vector3d(0.393358, -0.0362783, 0.0725647);
        Matrix3d ori_cur;
        Matrix3d ori_target; 
        ori_cur << 0.979439, 0.187027, 0.0756338,
        0.200865, -0.869147, -0.451926, 
        -0.0187854, 0.457826, -0.888844;

        ori_target << 0.0321804, -0.963101, -0.26721,
        -0.536335, -0.242236, 0.808496,
        -0.843392, 0.117296, -0.52434;


        cout << "entered reach" << endl;
        walle_pose_task->setGoalPosition(x_target);
        walle_pose_task->setGoalOrientation(ori_target);


        // update task model
        cout << "EXIT" << endl;
        walle_N_prec.setIdentity();
        walle_pose_task->updateTaskModel(walle_N_prec);
        walle_joint_task->updateTaskModel(walle_pose_task->getTaskAndPreviousNullspace());

        // compute torques
        cout << "command torques BEFORE" << "\n";
        cout << walle_command_torques << "\n";
        cout << "currentposition AFTER" << "\n";
        cout << walle_ee_pos << "\n";

        walle_command_torques = walle_joint_task->computeTorques() + walle_pose_task->computeTorques();

        cout << "command torques AFTER" << "\n";
        cout << walle_command_torques << "\n";
        cout << "currentposition AFTER" << "\n";
        cout << walle_ee_pos << "\n";

        

        // send to redis
        redis_client.sendAllFromGroup();
        cout << "sent it to redis" << endl;

        //redis_client.setEigen(JOINT_TORQUES_COMMANDED_WALLE_KEY, walle_command_torques);
        start = false;

    }

    cout << "reached outside" << endl;
    walle_command_torques.setZero();
    gravity_comp_enabled = true;
    redis_client.sendAllFromGroup();

    timer.stop();
    cout << "\nControl loop timer stats:\n";
    timer.printInfoPostRun();

    return 0;
}
