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
    POSTURE,
    MOTION,
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
const string robot_file = "${HW_FOLDER}/laundre/panda/panda_arm.urdf";
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

    // start redis client
    auto redis_client = Sai2Common::RedisClient();
    redis_client.connect();

    // set up signal handler
    signal(SIGABRT, &sighandler);
    signal(SIGTERM, &sighandler);
    signal(SIGINT, &sighandler);

    // keys 
    std::string MASSMATRIX_KEY = "sai2::FrankaPanda::Juliet::sensors::model::massmatrix"; 
	if (!flag_simulation) {
		JOINT_TORQUES_COMMANDED_WALLE_KEY = "sai2::FrankaPanda::Juliet::actuators::fgc";
		JOINT_ANGLES_WALLE_KEY = "sai2::FrankaPanda::Juliet::sensors::q";
		JOINT_VELOCITIES_WALLE_KEY = "sai2::FrankaPanda::Juliet::sensors::dq";
		//MASSMATRIX_KEY = "sai2::FrankaPanda::Juliet::sensors::model::massmatrix";  // NEED TO MAKE THIS
		//GRAVITY_COMP_ENABLED_KEY = "sai2::FrankaPanda::Juliet::sensors::model::robot_gravity";
	}

    // computer vision 
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
    walle->setQ(redis_client.getEigen(JOINT_ANGLES_WALLE_KEY));
    walle->setDq(redis_client.getEigen(JOINT_VELOCITIES_WALLE_KEY));
    walle->updateModel();

    std::cout << "Initial posture: \n" << redis_client.getEigen(JOINT_ANGLES_WALLE_KEY).transpose() << "\n";
    VectorXd q_init = walle->q();

    // prepare controller
    int dof = walle->dof(); // same as eve here
    VectorXd walle_command_torques = VectorXd::Zero(dof);  // panda + gripper torques 
    MatrixXd walle_N_prec = MatrixXd::Identity(dof, dof);

    // arm task (joints 0-6)
    const string control_link = "link7";
    const Vector3d control_point = Vector3d(0.195, -0.0, 0.0785);
    Affine3d compliant_frame = Affine3d::Identity();
    compliant_frame.translation() = control_point;

    auto walle_pose_task = std::make_shared<Sai2Primitives::MotionForceTask>(walle, control_link, compliant_frame);
    walle_pose_task->setPosControlGains(200, 20, 0);
    walle_pose_task->setOriControlGains(200, 20, 0);
    walle_pose_task->setDynamicDecouplingType(FULL_DYNAMIC_DECOUPLING);

    //walle_pose_task->enableInternalOtg();
    walle_pose_task->enableVelocitySaturation();

    Vector3d walle_ee_pos;
    Matrix3d walle_ee_ori;

    // joint task for posture control
    auto walle_joint_task = std::make_shared<Sai2Primitives::JointTask>(walle);
    walle_joint_task->enableVelocitySaturation(M_PI / 8);
    walle_joint_task->setGains(100, 20, 0);

    // double kp_j = 200.0;
    // double kv_j = 20.0;
    // walle_joint_task->setGains(kp_j, kv_j, 0);

    VectorXd q_desired(dof);
    q_desired <<-0.0590086,-0.265245,-0.0863773,-2.09088,-0.0457984,1.88314,-2.35595;
    // q_desired.head(7) << -30.0, -15.0, -15.0, -105.0, 0.0, 90.0, 45.0;
    // q_desired.head(7) *= M_PI / 180.0;
    // q_desired.tail(2) << 0.04, -0.04;
    walle_joint_task->setGoalPosition(q_desired);

    // flag for enabling gravity compensation
    bool gravity_comp_enabled = true;

    // zero out redis key command torques
    redis_client.setEigen(JOINT_TORQUES_COMMANDED_WALLE_KEY, walle_command_torques);

    // setup send and receive groups
    VectorXd walle_q = redis_client.getEigen(JOINT_ANGLES_WALLE_KEY);
    VectorXd walle_dq = redis_client.getEigen(JOINT_VELOCITIES_WALLE_KEY);
    MatrixXd mass_matrix = redis_client.getEigen(MASSMATRIX_KEY);

    // redis_client.addToReceiveGroup(JOINT_ANGLES_WALLE_KEY, walle_q);
    // redis_client.addToReceiveGroup(JOINT_VELOCITIES_WALLE_KEY, walle_dq);
    // redis_client.addToReceiveGroup(MASSMATRIX_KEY, mass_matrix);

    // redis_client.addToSendGroup(JOINT_TORQUES_COMMANDED_WALLE_KEY, walle_command_torques);
    //redis_client.addToSendGroup(GRAVITY_COMP_ENABLED_KEY, gravity_comp_enabled);

    // redis_client.receiveAllFromGroup();
    // redis_client.sendAllFromGroup();

    // update robot model from simulation configuration
    // walle->setQ(walle_q);
    // walle->setDq(walle_dq);    

    // if (!flag_simulation) {
    //     cout << "creating mass matrix" << endl;
    //     MatrixXd M = redis_client.getEigen(MASSMATRIX_KEY);
    //     // M(4, 4) += 0.2;
    //     // M(5, 5) += 0.2;
    //     // M(6, 6) += 0.2;
    //     walle->updateModel(M);
    // }

    // else {
    //     walle->updateModel();
    // }

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

    // state machine
    int state = POSTURE;

    while (runloop) {
        // wait for next scheduled loop
        timer.waitForNextLoop();
        double time = timer.elapsedSimTime();

        if (!flag_simulation) {
            walle_q = redis_client.getEigen(JOINT_ANGLES_WALLE_KEY);
            walle_dq = redis_client.getEigen(JOINT_VELOCITIES_WALLE_KEY);
        } else {
            walle_q = redis_client.getEigen(JOINT_ANGLES_WALLE_KEY);
            walle_dq = redis_client.getEigen(JOINT_VELOCITIES_WALLE_KEY);
        }

        // update robot 
        walle->setQ(walle_q);
        walle->setDq(walle_dq);

        if (!flag_simulation) {
            MatrixXd M = redis_client.getEigen(MASSMATRIX_KEY);
            // M(4, 4) += 0.2;
            // M(5, 5) += 0.2;
            // M(6, 6) += 0.2;
            walle->updateModel(M);
            // walle->updateModel();

        } else {
            walle->updateModel();
        }

//         // update kinematic targets
//         walle_ee_pos = walle->position(control_link, control_point);
//         walle_ee_ori = walle->rotation(control_link);
//         Matrix3d walle_R_desired = walle_ee_ori;
//         Vector3d walle_x_desired = walle_ee_pos;


//         Matrix3d ee_ori_world;
//         Matrix3d box_ori = Matrix3d::Zero();
//         Vector3d x_cur, x_target;

//         // cout << "start's value: " << start << endl;
//         // cout << "reach's value: " << REACH << endl;

// // 
//         // cout << "entered start" << endl;
//         x_cur =  Vector3d(-0.0423415, -0.611336, 0.425331);
//         // x_target = Vector3d(0.459529, 0.0571716, 0.358493);
//         x_target = Vector3d(0.1, 0.1, 0.1);
//         Matrix3d ori_cur;
//         Matrix3d ori_target; 
//         ori_cur <<  -0.80416, -0.355678,  0.476257,
//                     -0.513481,  0.819289, -0.255152,
//                     -0.29944, -0.449732, -0.841473;
//         ori_target << -0.80416, -0.355678,  0.476257,
//                     -0.513481,  0.819289, -0.255152,
//                     -0.29944, -0.449732, -0.841473;

        /*

        new one =  0.459529
        0.0571716
        0.358493
        -0.616993   0.655408  -0.435613
        0.726577   0.687069 0.00463211
        0.302333  -0.313649  -0.900122


        0.653312
        -0.00944733
        0.258136
        -0.552677  0.373262  0.745134
        0.638464  0.764289 0.0907009
        -0.535642  0.525869 -0.660719


        0.532921
        0.291814
        0.328841
        -0.855973 -0.0649666   0.512922
        0.33835   0.679746   0.650741
        -0.390933   0.730565  -0.559863
        
        */
    
        if (state == POSTURE) {

            q_desired(0) = q_init(0) + 1e-1 * sin(2 * M_PI * time);
            walle_joint_task->setGoalPosition(q_desired);

            // walle_N_prec.setIdentity();
            walle_joint_task->updateTaskModel(MatrixXd::Identity(7, 7));
            walle_command_torques = 1e0 * walle_joint_task->computeTorques();

            // if (walle_joint_task->goalPositionReached(1e-1)) {
            //     std::cout << "Posture to Motion\n";
            //     state = MOTION;
            //     continue;
            // }

        } else if (state == MOTION) {

            // cout << "entered reach" << endl;
            walle_pose_task->setGoalPosition(walle->position(control_link, control_point));
            //walle_pose_task->setGoalOrientation(ori_target);


            // update task model
            // cout << "EXIT" << endl;
            walle_N_prec.setIdentity();
            walle_pose_task->updateTaskModel(walle_N_prec);
            walle_joint_task->updateTaskModel(walle_pose_task->getTaskAndPreviousNullspace());

            // compute torques
            // cout << "command torques BEFORE" << "\n";
            // cout << walle_command_torques << "\n";
            // cout << "currentposition AFTER" << "\n";
            // cout << walle_ee_pos << "\n";

            walle_command_torques = 0 * walle_joint_task->computeTorques() + 0 * walle_pose_task->computeTorques();

        }

        // cout << "command torques AFTER" << "\n";
        cout << walle_command_torques.transpose() << "\n";
        // cout << "currentposition AFTER" << "\n";
        // cout << walle_ee_pos << "\n";

        

        // send to redis
        // redis_client.sendAllFromGroup();
        // cout << "sent it to redis" << endl;

        redis_client.setEigen(JOINT_TORQUES_COMMANDED_WALLE_KEY, walle_command_torques);
        start = false;

    }

    cout << "reached outside" << endl;
    walle_command_torques.setZero();

    // zero out redis key command torques
    redis_client.setEigen(JOINT_TORQUES_COMMANDED_WALLE_KEY, walle_command_torques);

    gravity_comp_enabled = true;
    // redis_client.sendAllFromGroup();

    timer.stop();
    cout << "\nControl loop timer stats:\n";
    timer.printInfoPostRun();

    return 0;
}
