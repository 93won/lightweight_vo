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
    
    bool shouldClose() const;
    void render();
    
    // Camera controls
    void setCameraPosition(float distance, float yaw, float pitch);
    void resetCamera();
    
    // 3D drawing functions
    void drawGrid();
    void drawOriginPoint();
    void drawAxis();
    void draw3DPoints(const std::vector<Eigen::Vector3f>& points);
    
    // Update functions
    void updatePoints(const std::vector<Eigen::Vector3f>& points);
    
    // Image display functions
    void updateTrackingImage(const cv::Mat& image);
    void updateStereoImage(const cv::Mat& image);
    GLuint createTextureFromMat(const cv::Mat& mat);
    
private:
    GLFWwindow* m_window;
    bool m_initialized;
    
    // 3D points to render
    std::vector<Eigen::Vector3f> m_points;
    
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
    
    // Image textures
    GLuint m_tracking_texture;
    GLuint m_stereo_texture;
    int m_tracking_width, m_tracking_height;
    int m_stereo_width, m_stereo_height;
    
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
