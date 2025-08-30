#pragma once

#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <Eigen/Dense>
#include <opencv2/opencv.hpp>
#include <memory>
#include <vector>

namespace lightweight_vio {

class ImGuiViewer {
public:
    ImGuiViewer();
    ~ImGuiViewer();

    bool initialize(int width = 2400, int height = 1200);
    void shutdown();
    
    bool should_close() const;
    bool is_ready() const;
    void render();
    
    // Camera controls
    void set_camera_position(float distance, float yaw, float pitch);
    void reset_camera();
    
    // 3D drawing functions
    void draw_grid();
    void draw_origin_point();
    void draw_axis();
    void draw_3d_points(const std::vector<Eigen::Vector3f>& points);
    void draw_pose(const Eigen::Matrix4f& pose);
    void draw_trajectory(const std::vector<Eigen::Vector3f>& trajectory);
    void draw_camera_frustum(const Eigen::Matrix4f& Twc);  // Add camera frustum drawing
    void draw_body_frame(const Eigen::Matrix4f& Twb);      // Add body frame drawing
    
    // Update functions
    void update_points(const std::vector<Eigen::Vector3f>& points);
    void update_pose(const Eigen::Matrix4f& pose);
    void update_trajectory(const std::vector<Eigen::Vector3f>& trajectory);
    void update_frame_with_transforms(const Eigen::Matrix4f& Twb);  // Add frame update with transforms
    
    // Image display functions
    void update_tracking_image(const cv::Mat& image);
    void update_stereo_image(const cv::Mat& image);
    GLuint create_texture_from_mat(const cv::Mat& mat);
    
    // Frame information update
    void update_frame_info(int frame_id, int total_features, int tracked_features, int new_features);
    
    // Input handling
    void process_keyboard_input(bool& auto_play, bool& step_mode, bool& advance_frame);
    
private:
    GLFWwindow* m_window;
    bool m_initialized;
    
    // 3D points to render
    std::vector<Eigen::Vector3f> m_points;
    
    // Current pose and trajectory
    Eigen::Matrix4f m_current_pose;
    std::vector<Eigen::Vector3f> m_trajectory;
    
    // Camera parameters
    float m_camera_distance;
    float m_camera_yaw;
    float m_camera_pitch;
    
    // Camera target (look-at point)
    Eigen::Vector3f m_camera_target;
    
    // Mouse control variables
    double m_last_mouse_x;
    double m_last_mouse_y;
    bool m_first_mouse;
    
        // Textures for image display
    GLuint m_tracking_texture;
    GLuint m_stereo_texture;
    int m_tracking_width, m_tracking_height;
    int m_stereo_width, m_stereo_height;
    
    // Keyboard input state
    bool m_space_pressed;
    bool m_next_pressed;
    
    // Frame information for VIO display
    int m_current_frame_id;
    int m_total_features;
    int m_tracked_features;
    int m_new_features;
    
    // OpenGL setup
    void setupOpenGL();
    void setupCamera();
    void setupProjection();
    
    // Mouse input processing
    void processCameraInput();
    
    // Event callbacks
    static void keyCallback(GLFWwindow* window, int key, int scancode, int action, int mods);
    static void scrollCallback(GLFWwindow* window, double xoffset, double yoffset);
};

} // namespace lightweight_vio
