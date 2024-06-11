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
#include "Sai2Graphics.h"
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
    BOTTOM_LEFT,
    RIGHT_SLEEVE,
    BOTTOM_RIGHT,
    BOTTOM,
    THIRD,
    THIRD2
};

enum Action {
    //VISION,
    REACH,
    SLIP,
    FLIP,
    RISE, 
    // LEFT,
    // RIGHT,
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
const string world_file = "${HW_FOLDER}/laundre/world.urdf";
const bool flag_simulation = false;

Matrix3d ori(Vector3d cur, Vector3d target) {
    Matrix3d result = Matrix3d::Zero();
    result(2, 2) = 1.0;
    Vector2d vec = (target - cur).normalized().head(2);
    result(0, 0) = vec(0);
    result(1, 0) = vec(1);
    result(0, 1) = -vec(1);
    result(1, 1) = vec(0);
    return result;
}

Matrix3d rotation(float angle) {
    // Matrix3d R;
    // R <<
    //     cos(M_PI*angle/180.0), 0.0, sin(M_PI*angle/180.0),
    //     0.0, 1.0, 0.0,
    //     -1*sin(M_PI*angle/180.0), 0.0, cos(M_PI*angle/180.0);
    return AngleAxisd(angle * (M_PI / 180), Vector3d::UnitY()).toRotationMatrix();
    // return R;
}

void displayRobot();

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
    //float platform_height = 0.249;
    // system("python3 /home/src0/sai2/apps/cs225a/homework/laundre/vision.py");
    // ifstream points_file("/home/src0/sai2/apps/cs225a/homework/laundre/points.txt"); //Calibrated xy coordinates in robot base frame
    // string num;
    // int points[8][2][2]; //Part / cur-target / x-y
    // int i=0, j=0, k=0;
    // if (points_file.is_open()) while(points_file) {
    //     cout << "A" << endl;
    //     points_file >> num;
    //     points[i][j][k] = stoi(num);

    //     k = 1-k;
    //     if (k==0) {
    //         j = 1-j;
    //         if (j==0) i += 1;
    //     }
    // }

    int testing = LEFT_SLEEVE;
    int part = testing;
    int action = 0;
    bool start = true;

    Sai2Model::URDF_FOLDERS["CS225A_URDF_FOLDER"] = string(CS225A_URDF_FOLDER);
    Sai2Model::URDF_FOLDERS["HW_FOLDER"] = string(HW_FOLDER);

    // load robots
    auto walle = std::make_shared<Sai2Model::Sai2Model>(robot_file);
    walle->setQ(redis_client.getEigen(JOINT_ANGLES_WALLE_KEY));
    walle->setDq(redis_client.getEigen(JOINT_VELOCITIES_WALLE_KEY));
    walle->updateModel();

    // std::cout << "Initial posture: \n" << redis_client.getEigen(JOINT_ANGLES_WALLE_KEY).transpose() << "\n";
    VectorXd q_init = walle->q();

    // prepare controller
    int dof = walle->dof(); // same as eve here
    VectorXd walle_command_torques = VectorXd::Zero(dof);  // panda + gripper torques 
    MatrixXd walle_N_prec = MatrixXd::Identity(dof, dof);

    // arm task (joints 0-6)
    const string control_link = "end-effector";
    const Vector3d control_point = Vector3d(0.0, 0.0, 0.0);
    Affine3d compliant_frame = Affine3d::Identity();
    compliant_frame.translation() = control_point;

    auto walle_pose_task = std::make_shared<Sai2Primitives::MotionForceTask>(walle, control_link, compliant_frame);
    walle_pose_task->setPosControlGains(200, 30, 0);
    walle_pose_task->setOriControlGains(200, 30, 0);
    // walle_pose_task->setDynamicDecouplingType(FULL_DYNAMIC_DECOUPLING);
    walle_pose_task->setDynamicDecouplingType(BOUNDED_INERTIA_ESTIMATES);

    //walle_pose_task->enableInternalOtg();
    walle_pose_task->enableVelocitySaturation();

    Vector3d walle_ee_pos;
    Matrix3d walle_ee_ori;

    // joint task for posture control
    auto walle_joint_task = std::make_shared<Sai2Primitives::JointTask>(walle);
    walle_joint_task->enableVelocitySaturation(M_PI / 3);
    walle_joint_task->setGains(500, 40, 20);
    walle_joint_task->setDynamicDecouplingType(BOUNDED_INERTIA_ESTIMATES);

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

    // start graphics thread
	thread key_read_thread(displayRobot);

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
            walle->updateModel(M);

        } else {
            walle->updateModel();
        }

        // update kinematic targets
        walle_ee_pos = walle->position(control_link, control_point);
        walle_ee_ori = walle->rotation(control_link);
        Matrix3d walle_R_desired = walle_ee_ori;

        Vector3d walle_x_desired = walle_ee_pos;

        // curr = walle end effector position:  0.437749 -0.348901  0.263422
        // des = walle end effector position:  0.475617 -0.228679  0.259935

        Matrix3d ee_ori_world;
        // Matrix3d box_ori = Matrix3d::Zero();
        Vector3d x_cur, x_target;
        // cout << "walle end effector position: " << walle_ee_pos.transpose() << endl;
        float approach_angle, slip_angle, end_angle;
        Vector3d offset;
        Vector3d direction;
        //cout << "---" << endl << walle_ee_pos << "===" << endl;
        //cout << "---" << endl << walle_q << "===" << endl;
    
        if (start) {
            cout << part << endl;
            // if (part == testing) {
            //     // WAS FOR RIGHT SLEVE
            //     x_cur = Vector3d(0.3, 0.4, platform_height);
            //     x_target = Vector3d(0.2, 0.25, platform_height);

            //     // x_cur = Vector3d(0.45, -0.15, platform_height);
            //     // x_target = Vector3d(0.3, -0.1, platform_height);
            //     // x_cur = Vector3d(points[part][0][0], points[part][0][0], platform_height);
            //     // x_target = Vector3d(points[part][1][0], points[part][1][0], platform_height);
            //     approach_angle = 45.0;
            //     slip_angle = 20.0;
            //     end_angle = 0.0;
            // }

            if (part == LEFT_SLEEVE) {
                direction = Vector3d(0.0, 0.3, 0.0);
                //q_desired <<-0.934917, 0.564647,-0.187367,-2.59346,0.222665,1.88575,0.953446;
                q_desired << -1.15034,0.0852058,0.0198756,-2.2224,-0.0879379,2.34664,1.3072;
                approach_angle = 90.0;
                slip_angle = 5.0;
                end_angle = 10.0;
                offset = Vector3d(0.0, 0.0, 0.2);
            } else if (part == BOTTOM_LEFT) {
                direction = Vector3d(-0.2, 0.2, 0.0);
                q_desired <<-0.610156,0.385604,0.158066,-1.76931,-0.266216,2.31821,1.30639;
                approach_angle = 90.0;
                slip_angle = 20.0;
                end_angle = 0.0;
                offset = Vector3d(0.0, 0.0, 0.2);
            } else if (part == RIGHT_SLEEVE) {
                direction = Vector3d(0.0, -0.2, 0.0);
                q_desired <<0.868574,0.594035,0.349277,-1.53278,-0.415279,1.6789,0.560107;
                approach_angle = 10.0;
                slip_angle = 0.0;
                end_angle = 10.0;
                offset = Vector3d(0.0, 0.0, 0.2);
            } else if (part == BOTTOM_RIGHT) {
                direction = Vector3d(-0.2, -0.2, 0.0);
                q_desired <<0.509472,0.995239,-0.201277,-.784185,0.153668,1.70496,0.494889;
                approach_angle = 90.0;
                slip_angle = 20.0;
                end_angle = 0.0;
                offset = Vector3d(0.0, 0.0, 0.2);
            } else if (part == BOTTOM) {
                direction = Vector3d(-0.2, 0.0, 0.0);
                q_desired <<0.231137,0.8175,-0.403637,-1.15099,0.256269,1.84777,0.640275;
                approach_angle = 70.0;
                slip_angle = 20.0;
                end_angle = 100.0;
                offset = Vector3d(0.0, 0.0, 0.0);
            } else if (part == THIRD) {
                direction = Vector3d(0.0, 0.2, 0.0);
                q_desired <<-0.392532,0.568462,-0.800611,-1.90843,0.528432,1.6917,0.783055;
                approach_angle = 10.0;
                slip_angle = 0.0;
                end_angle = 100.0;
                offset = Vector3d(0.0, 0.0, 0.0);
            } else if (part == THIRD2) {
                direction = Vector3d(0.0, 0.2, 0.0);
                q_desired <<-0.562196,0.172495,-0.474372,-2.32864,0.383207,1.91946,1.08381;
                approach_angle = 10.0;
                slip_angle = 0.0;
                end_angle = 100.0;
                offset = Vector3d(0.0, 0.0, 0.0);
            }

            if (action == REACH) {
                cout << "REACH" << endl;
                cout << q_desired.transpose() << endl;
                walle_joint_task->setGoalPosition(q_desired);
                // walle_pose_task->setGoalPosition(0.0, 0)

            } 
            else if (action == SLIP) {
                cout << "SLIP" << endl;

                walle_pose_task->parametrizeForceMotionSpaces(
					1, Vector3d::UnitZ());

				// set the force control
				walle_pose_task->setGoalForce(-2.0 * Vector3d::UnitZ());

				walle_pose_task->setForceControlGains(35.0, 20.0, 0.0);
                // walle_pose_task->setGoalPosition(x_target);
                walle_pose_task->setGoalPosition(walle_ee_pos + direction);
                walle_pose_task->setGoalOrientation(ori(Vector3d::Zero(), direction) * rotation(slip_angle));
                // walle_pose_task->setGoalOrientation(AngleAxisd(M_PI / 2, Vector3d::UnitZ()).toRotationMatrix() * rotation(end_angle));

            } else if (action == FLIP) {
                cout << "FLIP" << endl;

                // Keep end effector position fixed, only change orientation
                walle_pose_task->setGoalPosition(walle_ee_pos + offset);
                walle_pose_task->parametrizeForceMotionSpaces(0);
                walle_pose_task->setGoalOrientation(ori(Vector3d::Zero(), direction)* rotation(end_angle) );
                // walle_pose_task->setGoalOrientation(AngleAxisd(M_PI / 2, Vector3d::UnitZ()).toRotationMatrix() * rotation(end_angle));
            } else if (action == RISE) {
                cout << "RISE" << endl;
                //walle_pose_task->parametrizeForceMotionSpaces(0);
                walle_pose_task->setGoalPosition(walle_ee_pos + Vector3d(0, 0, 0.2));
                walle_pose_task->setGoalOrientation(ori(Vector3d::Zero(), direction) * rotation(90.0));
            }

            start = false;
            if (part <= BOTTOM_LEFT) {
                cout << "Working on action " << action << " with part " << part << "\n";
            } else {
                cout << "Done!" << "\n";
            }
        } else if (part <= BOTTOM_LEFT &&
            ((action == REACH && walle_joint_task->goalPositionReached(0.15)) ||
            (action != REACH && walle_pose_task->goalPositionReached(0.07) &&
            walle_pose_task->goalOrientationReached(0.10))))
        {
            cout << "Moving on! Yay!" << "\n";
            action = (action + 1) % 4;
            cout << "Action: " << action << endl;

            start = true;
            if (action == 0) part += 1;
            
            //if (action == 0) return 0;
            if (part > BOTTOM_LEFT) {
                break;
            }
        } else {
            // cout << "--" << q_desired - walle_q << "==" << endl;
        }

        // print out orientation
        // std::cout << "Robot orientation:\n" << walle->rotation("end-effector") << "\n";
        // std::cout << "Desired orientation:\n" << walle_pose_task->getGoalOrientation() << "\n";

        // update task model
        walle_N_prec.setIdentity();

        // compute torques
        if (action == REACH) {
            walle_joint_task->updateTaskModel(walle_N_prec);
            walle_command_torques = walle_joint_task->computeTorques();
            // std::cout << "Reach state\n";
            // std::cout << "Q desired \n" << q_desired.transpose() << "\n";
            // std::cout << "Q current \n" << walle->q().transpose() << "\n";
        }
        else {
            // std::cout << "Pose state\n";
            walle_pose_task->updateTaskModel(walle_N_prec);
            walle_joint_task->updateTaskModel(walle_pose_task->getTaskAndPreviousNullspace());
            walle_command_torques = walle_pose_task->computeTorques() + walle_joint_task->computeTorques();
        }

        // cout << walle_command_torques << endl;

        //walle_command_torques = walle_joint_task->computeTorques()+walle_pose_task->computeTorques();

        redis_client.setEigen(JOINT_TORQUES_COMMANDED_WALLE_KEY, 1 * walle_command_torques);
    }

    cout << "reached outside" << endl;
    walle_command_torques.setZero();

    // zero out redis key command torques
    redis_client.setEigen(JOINT_TORQUES_COMMANDED_WALLE_KEY, 0 * walle_command_torques);

    gravity_comp_enabled = true;
    // redis_client.sendAllFromGroup();

    timer.stop();
    cout << "\nControl loop timer stats:\n";
    timer.printInfoPostRun();

    return 0;
}

//------------------------------------------------------------------------------
void displayRobot() {

	auto redis_client = Sai2Common::RedisClient();
	redis_client.connect();

	auto graphics = std::make_shared<Sai2Graphics::Sai2Graphics>(world_file);
	graphics->setBackgroundColor(66.0/255, 135.0/255, 245.0/255);
	graphics->showLinkFrame(true, "WALLE", "link0", 0.15);	    
	graphics->showLinkFrame(true, "WALLE", "end-effector", 0.15);	

	// graphics timer
	Sai2Common::LoopTimer graphicsTimer(30.0, 1e6);

	while (graphics->isWindowOpen()) {
		graphicsTimer.waitForNextLoop();

        // redis_client.get

		// for (auto& key : key_pressed) {
		// 	key_pressed[key.first] = graphics->isKeyPressed(key.first);
		// }

		// VectorXd robot_q = VectorXd::Zero(9);
		// robot_q.head(7) = redis_client.getEigen(JOINT_ANGLES_KEY);
		// double gripper_width = redis_client.getDouble(GRIPPER_CURRENT_WIDTH_KEY);		
		// robot_q(7) = gripper_width / 2;
		// robot_q(8) = - gripper_width / 2;
		graphics->updateRobotGraphics("WALLE", redis_client.getEigen(JOINT_ANGLES_WALLE_KEY));

		// Vector3d proxy_pos = redis_client.getEigen(PROXY_POS_KEY);
		// Matrix3d proxy_ori = redis_client.getEigen(PROXY_ORI_KEY);
		// Affine3d proxy_transform;
		// proxy_transform.translation() = proxy_pos;
		// proxy_transform.linear() = proxy_ori;
		// graphics->updateObjectGraphics("proxy_end_effector", proxy_transform);

		graphics->renderGraphicsWorld();
	}

	runloop = false;
}