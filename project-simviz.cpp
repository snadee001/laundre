#include "simviz/SimVizRedisInterface.h"
#include "simviz/SimVizConfigParser.h"

int main(int argc, char** argv) {
	Sai2Model::URDF_FOLDERS["CS225A_URDF_FOLDER"] = std::string(CS225A_URDF_FOLDER);
	Sai2Model::URDF_FOLDERS["PROJECT_FOLDER"] = std::string(HW_FOLDER) + "/laundre";
	std::string config_file = std::string(HW_FOLDER) + "/laundre/simviz_config.xml";
	Sai2Interfaces::SimVizConfigParser parser;
	Sai2Interfaces::SimVizRedisInterface simviz(parser.parseConfig(config_file));
	simviz.run();
	return 0;
}
