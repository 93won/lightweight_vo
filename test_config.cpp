#include <iostream>
#include "src/util/Config.h"

using namespace lightweight_vio;

int main() {
    std::cout << "=== Config Test ===" << std::endl;
    
    try {
        Config::getInstance().load("../config/euroc.yaml");
        std::cout << "Configuration loaded successfully" << std::endl;
        
        // Test some config values
        std::cout << "Max features: " << Config::getInstance().getMaxFeatures() << std::endl;
        std::cout << "Image width: " << Config::getInstance().getImageWidth() << std::endl;
        std::cout << "Image height: " << Config::getInstance().getImageHeight() << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "Failed to load configuration: " << e.what() << std::endl;
        return -1;
    }
    
    return 0;
}
