#include "ImGuiViewer.h"
#include <iostream>
#include <cmath>
#include <GL/glu.h>

#include "../../thirdparty/imgui/imgui.h"
#include "../../thirdparty/imgui/backends/imgui_impl_glfw.h"
#include "../../thirdparty/imgui/backends/imgui_impl_opengl2.h"

namespace lightweight_vio {

ImGuiViewer::ImGuiViewer()
    : m_window(nullptr)
    , m_initialized(false)
    , m_camera_distance(10.0f)
    , m_camera_yaw(0.0f)           // Store in degrees like original
    , m_camera_pitch(-17.0f)       // Store in degrees like original (-0.3 rad ≈ -17 deg)
    , m_camera_target(0.0f, 0.0f, 0.0f)
    , m_last_mouse_x(0.0)
    , m_last_mouse_y(0.0)
    , m_first_mouse(true)
    , m_tracking_texture(0)
    , m_stereo_texture(0)
    , m_tracking_width(0)
    , m_tracking_height(0)
    , m_stereo_width(0)
    , m_stereo_height(0)
{
}

ImGuiViewer::~ImGuiViewer() {
    shutdown();
}

bool ImGuiViewer::initialize(int width, int height) {
    // Initialize GLFW
    if (!glfwInit()) {
        std::cerr << "Failed to initialize GLFW" << std::endl;
        return false;
    }

    // Create window
    m_window = glfwCreateWindow(width, height, "ImGui 3D Viewer", nullptr, nullptr);
    if (!m_window) {
        std::cerr << "Failed to create GLFW window" << std::endl;
        glfwTerminate();
        return false;
    }

    glfwMakeContextCurrent(m_window);
    glfwSwapInterval(1); // Enable vsync

    // Initialize GLEW
    if (glewInit() != GLEW_OK) {
        std::cerr << "Failed to initialize GLEW" << std::endl;
        glfwDestroyWindow(m_window);
        glfwTerminate();
        return false;
    }

    // Set callbacks
    glfwSetWindowUserPointer(m_window, this);
    glfwSetKeyCallback(m_window, keyCallback);
    glfwSetScrollCallback(m_window, scrollCallback);

    // Setup ImGui
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO& io = ImGui::GetIO(); (void)io;
    io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;

    ImGui::StyleColorsDark();

    ImGui_ImplGlfw_InitForOpenGL(m_window, true);
    ImGui_ImplOpenGL2_Init();

    // Setup OpenGL
    setupOpenGL();

    m_initialized = true;
    std::cout << "ImGui 3D Viewer initialized successfully" << std::endl;
    return true;
}

void ImGuiViewer::shutdown() {
    if (m_initialized) {
        // Clean up textures
        if (m_tracking_texture) {
            glDeleteTextures(1, &m_tracking_texture);
            m_tracking_texture = 0;
        }
        if (m_stereo_texture) {
            glDeleteTextures(1, &m_stereo_texture);
            m_stereo_texture = 0;
        }
        
        ImGui_ImplOpenGL2_Shutdown();
        ImGui_ImplGlfw_Shutdown();
        ImGui::DestroyContext();
    }

    if (m_window) {
        glfwDestroyWindow(m_window);
        m_window = nullptr;
    }
    
    glfwTerminate();
    m_initialized = false;
}

bool ImGuiViewer::shouldClose() const {
    return m_window ? glfwWindowShouldClose(m_window) : true;
}

void ImGuiViewer::render() {
    if (!m_initialized || !m_window) return;

    glfwPollEvents();
    
    // Process camera input
    processCameraInput();

    // Start ImGui frame
    ImGui_ImplOpenGL2_NewFrame();
    ImGui_ImplGlfw_NewFrame();
    ImGui::NewFrame();

    // Get framebuffer size
    int display_w, display_h;
    glfwGetFramebufferSize(m_window, &display_w, &display_h);
    
    // Setup viewport and clear
    glViewport(0, 0, display_w, display_h);
    glClearColor(0.1f, 0.1f, 0.15f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    // Setup 3D rendering
    setupProjection();
    setupCamera();

    // Draw 3D content
    drawGrid();
    drawAxis();
    drawOriginPoint();
    draw3DPoints(m_points);

    // ImGui UI
    ImGui::Begin("3D Viewer Controls");
    ImGui::Text("Camera Controls:");
    ImGui::SliderFloat("Distance", &m_camera_distance, 1.0f, 50.0f);
    ImGui::SliderFloat("Yaw (deg)", &m_camera_yaw, -180.0f, 180.0f);
    ImGui::SliderFloat("Pitch (deg)", &m_camera_pitch, -89.0f, 89.0f);
    ImGui::SliderFloat3("Target", m_camera_target.data(), -10.0f, 10.0f);
    if (ImGui::Button("Reset Camera")) {
        resetCamera();
    }
    ImGui::Separator();
    ImGui::Text("3D Points: %zu", m_points.size());
    ImGui::Text("Mouse Controls:");
    ImGui::Text("  Left Click + Drag: Rotate camera");
    ImGui::Text("  Right Click + Drag: Pan camera");
    ImGui::Text("  Scroll Wheel: Zoom in/out");
    ImGui::Text("Keyboard:");
    ImGui::Text("  ESC: Exit application");
    ImGui::End();

    // Feature Tracking Image Window
    if (m_tracking_texture) {
        ImGui::Begin("Feature Tracking", nullptr, ImGuiWindowFlags_AlwaysAutoResize);
        ImGui::Image((void*)(intptr_t)m_tracking_texture, 
                    ImVec2(m_tracking_width * 1.0f, m_tracking_height * 1.0f));
        ImGui::End();
    }
    
    // Stereo Matching Image Window  
    if (m_stereo_texture) {
        ImGui::Begin("Stereo Matching", nullptr, ImGuiWindowFlags_AlwaysAutoResize);
        ImGui::Image((void*)(intptr_t)m_stereo_texture, 
                    ImVec2(m_stereo_width * 1.0f, m_stereo_height * 1.0f));
        ImGui::End();
    }

    // Render ImGui
    ImGui::Render();
    ImGui_ImplOpenGL2_RenderDrawData(ImGui::GetDrawData());

    glfwSwapBuffers(m_window);
}

void ImGuiViewer::setCameraPosition(float distance, float yaw, float pitch) {
    m_camera_distance = distance;
    m_camera_yaw = yaw;
    m_camera_pitch = pitch;
}

void ImGuiViewer::resetCamera() {
    m_camera_distance = 10.0f;
    m_camera_yaw = 0.0f;       // degrees
    m_camera_pitch = -17.0f;   // degrees  
    m_camera_target = Eigen::Vector3f(0.0f, 0.0f, 0.0f);
}

void ImGuiViewer::drawGrid() {
    glDisable(GL_DEPTH_TEST);
    glLineWidth(1.0f);
    glBegin(GL_LINES);
    
    float grid_size = 10.0f;
    float step = 1.0f;
    
    // Grid lines
    for (float i = -grid_size; i <= grid_size; i += step) {
        // X direction lines (parallel to Y axis)
        if (std::abs(i) < 0.001f) {
            glColor3f(0.0f, 1.0f, 0.0f); // Y axis in green
        } else {
            glColor3f(0.3f, 0.3f, 0.3f); // Grid lines in gray
        }
        glVertex3f(i, -grid_size, 0.0f);
        glVertex3f(i, grid_size, 0.0f);
        
        // Y direction lines (parallel to X axis)
        if (std::abs(i) < 0.001f) {
            glColor3f(1.0f, 0.0f, 0.0f); // X axis in red
        } else {
            glColor3f(0.3f, 0.3f, 0.3f); // Grid lines in gray
        }
        glVertex3f(-grid_size, i, 0.0f);
        glVertex3f(grid_size, i, 0.0f);
    }
    
    glEnd();
    glEnable(GL_DEPTH_TEST);
}

void ImGuiViewer::drawOriginPoint() {
    glPointSize(10.0f);
    glBegin(GL_POINTS);
    glColor3f(1.0f, 0.0f, 0.0f); // Red color
    glVertex3f(0.0f, 0.0f, 0.0f); // Origin
    glEnd();
}

void ImGuiViewer::drawAxis() {
    glLineWidth(3.0f);
    glBegin(GL_LINES);
    
    // X axis - Red
    glColor3f(1.0f, 0.0f, 0.0f);
    glVertex3f(0.0f, 0.0f, 0.0f);
    glVertex3f(2.0f, 0.0f, 0.0f);
    
    // Y axis - Green
    glColor3f(0.0f, 1.0f, 0.0f);
    glVertex3f(0.0f, 0.0f, 0.0f);
    glVertex3f(0.0f, 2.0f, 0.0f);
    
    // Z axis - Blue
    glColor3f(0.0f, 0.0f, 1.0f);
    glVertex3f(0.0f, 0.0f, 0.0f);
    glVertex3f(0.0f, 0.0f, 2.0f);
    
    glEnd();
    glLineWidth(1.0f);
}

void ImGuiViewer::draw3DPoints(const std::vector<Eigen::Vector3f>& points) {
    if (points.empty()) return;
    
    glPointSize(3.0f);
    glBegin(GL_POINTS);
    
    // Draw points in cyan color
    glColor3f(0.0f, 1.0f, 1.0f);
    for (const auto& point : points) {
        glVertex3f(point.x(), point.y(), point.z());
    }
    
    glEnd();
    glPointSize(1.0f);
}

void ImGuiViewer::updatePoints(const std::vector<Eigen::Vector3f>& points) {
    m_points = points;
}

void ImGuiViewer::setupOpenGL() {
    glEnable(GL_DEPTH_TEST);
    glDepthFunc(GL_LESS);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
}

void ImGuiViewer::setupCamera() {
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
    
    // Calculate camera position using spherical coordinates (like original viewer)
    float yaw_rad = m_camera_yaw * M_PI / 180.0f;      // Convert degrees to radians
    float pitch_rad = m_camera_pitch * M_PI / 180.0f;  // Convert degrees to radians
    
    float x = m_camera_target.x() - m_camera_distance * cos(pitch_rad) * cos(yaw_rad);
    float y = m_camera_target.y() - m_camera_distance * sin(pitch_rad);
    float z = m_camera_target.z() - m_camera_distance * cos(pitch_rad) * sin(yaw_rad);
    
    // Look at target point (up vector changed to match original)
    gluLookAt(x, y, z,                                    // Camera position
              m_camera_target.x(), m_camera_target.y(), m_camera_target.z(),  // Look at point (target)
              0, -1, 0);                                   // Up vector (back to original -1)
}

void ImGuiViewer::setupProjection() {
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    
    int width, height;
    glfwGetFramebufferSize(m_window, &width, &height);
    float aspect = static_cast<float>(width) / static_cast<float>(height);
    
    gluPerspective(45.0, aspect, 0.1, 100.0);
}

void ImGuiViewer::keyCallback(GLFWwindow* window, int key, int scancode, int action, int mods) {
    if (key == GLFW_KEY_ESCAPE && action == GLFW_PRESS) {
        glfwSetWindowShouldClose(window, GLFW_TRUE);
    }
}

void ImGuiViewer::processCameraInput() {
    ImGuiIO& io = ImGui::GetIO();
    if (io.WantCaptureMouse) {
        glfwGetCursorPos(m_window, &m_last_mouse_x, &m_last_mouse_y); // Update last pos to prevent jumps
        return;
    }

    double current_x, current_y;
    glfwGetCursorPos(m_window, &current_x, &current_y);

    if (glfwGetMouseButton(m_window, GLFW_MOUSE_BUTTON_LEFT) == GLFW_PRESS) {
        m_camera_yaw -= (current_x - m_last_mouse_x) * 0.25f;      // Back to original direction like the reference code
        m_camera_pitch -= (current_y - m_last_mouse_y) * 0.25f;   // Back to original direction like the reference code
        m_camera_pitch = std::max(-89.0f, std::min(89.0f, m_camera_pitch));
    } 
    else if (glfwGetMouseButton(m_window, GLFW_MOUSE_BUTTON_RIGHT) == GLFW_PRESS) {
        // Pan camera target (same as original viewer logic)
        float yaw_rad = m_camera_yaw * M_PI / 180.0f;
        float pitch_rad = m_camera_pitch * M_PI / 180.0f;
        Eigen::Vector3f eye, front;
        front.x() = cos(yaw_rad) * cos(pitch_rad);
        front.y() = sin(pitch_rad);
        front.z() = sin(yaw_rad) * cos(pitch_rad);
        eye = m_camera_target - m_camera_distance * front;
        
        Eigen::Vector3f view_dir = (m_camera_target - eye).normalized();
        Eigen::Vector3f world_up(0.0f, -1.0f, 0.0f);
        Eigen::Vector3f right = world_up.cross(view_dir).normalized();
        Eigen::Vector3f up = view_dir.cross(right).normalized();

        float dx = (current_x - m_last_mouse_x) * 0.001f * m_camera_distance;
        float dy = (current_y - m_last_mouse_y) * 0.001f * m_camera_distance;

        m_camera_target += right * dx;
        m_camera_target += up * dy;
    }

    m_last_mouse_x = current_x;
    m_last_mouse_y = current_y;
}

void ImGuiViewer::scrollCallback(GLFWwindow* window, double xoffset, double yoffset) {
    ImGuiViewer* viewer = static_cast<ImGuiViewer*>(glfwGetWindowUserPointer(window));
    ImGuiIO& io = ImGui::GetIO();
    if (viewer && !io.WantCaptureMouse) {
        viewer->m_camera_distance -= yoffset * 2.0f;  // Changed from 1.0f to 2.0f to match original
        if (viewer->m_camera_distance < 1.0f) {
            viewer->m_camera_distance = 1.0f;
        }
        if (viewer->m_camera_distance > 100.0f) {
            viewer->m_camera_distance = 100.0f;
        }
    }
}

void ImGuiViewer::updateTrackingImage(const cv::Mat& image) {
    if (image.empty()) return;
    
    // Convert to RGB if needed
    cv::Mat rgb_image;
    if (image.channels() == 1) {
        cv::cvtColor(image, rgb_image, cv::COLOR_GRAY2RGB);
    } else if (image.channels() == 3) {
        cv::cvtColor(image, rgb_image, cv::COLOR_BGR2RGB);
    } else {
        rgb_image = image;
    }
    
    // Delete old texture
    if (m_tracking_texture) {
        glDeleteTextures(1, &m_tracking_texture);
    }
    
    // Create new texture
    m_tracking_texture = createTextureFromMat(rgb_image);
    m_tracking_width = rgb_image.cols;
    m_tracking_height = rgb_image.rows;
}

void ImGuiViewer::updateStereoImage(const cv::Mat& image) {
    if (image.empty()) return;
    
    // Convert to RGB if needed
    cv::Mat rgb_image;
    if (image.channels() == 1) {
        cv::cvtColor(image, rgb_image, cv::COLOR_GRAY2RGB);
    } else if (image.channels() == 3) {
        cv::cvtColor(image, rgb_image, cv::COLOR_BGR2RGB);
    } else {
        rgb_image = image;
    }
    
    // Delete old texture
    if (m_stereo_texture) {
        glDeleteTextures(1, &m_stereo_texture);
    }
    
    // Create new texture
    m_stereo_texture = createTextureFromMat(rgb_image);
    m_stereo_width = rgb_image.cols;
    m_stereo_height = rgb_image.rows;
}

GLuint ImGuiViewer::createTextureFromMat(const cv::Mat& mat) {
    GLuint texture;
    glGenTextures(1, &texture);
    glBindTexture(GL_TEXTURE_2D, texture);
    
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, mat.cols, mat.rows, 0, GL_RGB, GL_UNSIGNED_BYTE, mat.data);
    
    return texture;
}

} // namespace lightweight_vio
