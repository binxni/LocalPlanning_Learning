#include "path_follower_pkg/path_follower.hpp"
#include <csignal>
#include <memory>

// Global pointer to the node for signal handler
std::shared_ptr<path_follower_pkg::PathFollower> g_node = nullptr;

void signalHandler(int /* signum */) {
    if (g_node) {
        g_node->shutdown_handler();
    }
    rclcpp::shutdown();
}

int main(int argc, char** argv) {
    rclcpp::init(argc, argv);
    
    auto node = std::make_shared<path_follower_pkg::PathFollower>();
    g_node = node;  // Set global pointer for signal handler
    
    // Register signal handlers
    std::signal(SIGINT, signalHandler);   // Ctrl+C
    std::signal(SIGTERM, signalHandler);  // Termination signal
    
    RCLCPP_INFO(node->get_logger(), "Starting Path Follower Node with graceful shutdown");
    
    rclcpp::spin(node);
    
    rclcpp::shutdown();
    return 0;
}