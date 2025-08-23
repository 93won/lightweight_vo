#!/bin/bash

# Create a simple test to see the logs
cd /home/eugene/lightweight_vio/build

# Create dummy config test
cat > test_config_simple.cpp << 'EOF'
#include <iostream>
#include "../src/util/Config.h"

using namespace lightweight_vio;

int main() {
    std::cout << "=== Config Load Test ===" << std::endl;
    
    try {
        Config::getInstance().load("../config/euroc.yaml");
        std::cout << "Configuration loaded successfully" << std::endl;
        std::cout << "Max features: " << Config::getInstance().getMaxFeatures() << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
    }
    
    return 0;
}
EOF

echo "Created test file"
