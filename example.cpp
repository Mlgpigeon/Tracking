#include <stdio.h>
#define _USE_MATH_DEFINES
#include <math.h>
#include "aruco.h"
#include "cvdrawingutils.h"
#include <fstream>
#include <iostream>
#include <stack>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/calib3d/calib3d.hpp>

#include <sstream>
#include <string>
#include <stdexcept>

#include <glad/glad.h>
#include <GLFW/glfw3.h>

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <glm/gtx/string_cast.hpp>
#include <glm/gtx/transform.hpp>
#include <glm/gtc/constants.hpp>
#include <glm/gtx/rotate_vector.hpp>

#include "shader.h"
#include "opengl_tools.h"

void framebuffer_size_callback(GLFWwindow* window, int width, int height);
void processInput(GLFWwindow *window);

// settings
const unsigned int SCR_WIDTH = 1280;
const unsigned int SCR_HEIGHT = 720;

#ifdef _WIN32
std::string BASE_PATH = "C:/Users/Luis/Desktop/Master/Realidad_Virtual/rv_practica_tracking_alumno/";
#else
std::string BASE_PATH = "/Users/dancasas/Work/dev/realidad_virtual_2020/rv_practica_tracking/";
#endif

// aruco
float f = 0.0;

aruco::MarkerDetector MDetector;
cv::VideoCapture TheVideoCapturer;
std::vector<aruco::Marker> TheMarkers;
cv::Mat TheInputImage, TheInputImageGrey, TheInputImageCopy;
aruco::CameraParameters TheCameraParameters;
int iDetectMode = 0, iMinMarkerSize = 0, iCorrectionRate = 0, iShowAllCandidates = 0, iEnclosed = 0, iThreshold, iCornerMode, iDictionaryIndex, iTrack = 0;

class CmdLineParser { int argc; char** argv; public:CmdLineParser(int _argc, char** _argv) : argc(_argc), argv(_argv) {}   bool operator[](std::string param) { int idx = -1;  for (int i = 0; i < argc && idx == -1; i++)if (std::string(argv[i]) == param)idx = i; return (idx != -1); }    std::string operator()(std::string param, std::string defvalue = "-1") { int idx = -1; for (int i = 0; i < argc && idx == -1; i++)if (std::string(argv[i]) == param)idx = i; if (idx == -1)return defvalue; else return (argv[idx + 1]); } };
struct   TimerAvrg { std::vector<double> times; size_t curr = 0, n; std::chrono::high_resolution_clock::time_point begin, end;   TimerAvrg(int _n = 30) { n = _n; times.reserve(n); }inline void start() { begin = std::chrono::high_resolution_clock::now(); }inline void stop() { end = std::chrono::high_resolution_clock::now(); double duration = double(std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count())*1e-6; if (times.size() < n) times.push_back(duration); else { times[curr] = duration; curr++; if (curr >= times.size()) curr = 0; } }double getAvrg() { double sum = 0; for (auto t : times) sum += t; return sum / double(times.size()); } };
TimerAvrg Fps;

// camera viewpoint
glm::vec3 current_pos = glm::vec3(-0.03f, -0.05f, -0.20f);
std::queue<glm::vec3> previous_poses;
std::queue<cv::Mat> previous_rvec;
float data_orientation[10] = { 0,1,0 };
glm::vec4 orientation = glm::vec4(0.0, 1.0, 0.0, 1.0);
glm::vec3 light_position = glm::vec3(1.0f, 1.0f, 1.0f);
glm::vec3 current_pos_left = glm::vec3(0.2f, 0.2f, 0.5f);
glm::vec3 current_pos_right = glm::vec3(0.2f, 0.2f, 0.5f);
glm::vec3 object_hand = glm::vec3(0.0f, 0.0f, 0.0f);

// converts cv::Mat of size 3x3 to glm::mat3
bool fromCV2GLM_3x3(const cv::Mat& cvmat, glm::mat3* glmmat) {
    if (cvmat.cols != 3 || cvmat.rows != 3 || cvmat.type() != CV_32FC1) {
        std::cout << "Matrix conversion error!" << std::endl;
        return false;
    }
    memcpy(glm::value_ptr(*glmmat), cvmat.data, 9 * sizeof(float));
    *glmmat = glm::transpose(*glmmat);
    return true;
}

// converts glm::mat3 to cv::Mat of size 3x3
bool fromGLM2CV_3x3(const glm::mat3& glmmat, cv::Mat* cvmat) {
    if (cvmat->cols != 3 || cvmat->rows != 3) {
        (*cvmat) = cv::Mat(3, 3, CV_32F);
    }
    memcpy(cvmat->data, glm::value_ptr(glmmat), 9 * sizeof(float));
    *cvmat = cvmat->t();
    return true;
}

//
// computes the mean of a std::queue<cv::Mat>. Used to average the last few camera rotations to smooth trajectory
//
cv::Mat compute_mean_cvMat(std::queue<cv::Mat> current_queue, int NUMBER_OF_FRAMES_TO_AVERAGE, bool VERBOSE = false)
{
    std::queue<cv::Mat>current_queue_copy = current_queue;
    cv::Mat mean_current_queue(cv::Size(1, 3), CV_32F);

    mean_current_queue.at<float>(0, 0) = 0.0f;
    mean_current_queue.at<float>(0, 1) = 0.0f;
    mean_current_queue.at<float>(0, 2) = 0.0f;

    if (VERBOSE)
        printf("Queue status: ");

    while (!current_queue_copy.empty())
    {
        if (VERBOSE)
        {
            printf("(% .02f ", current_queue_copy.front().at<float>(0, 0));
            printf("% .02f ", current_queue_copy.front().at<float>(0, 1));
            printf("% .02f), ", current_queue_copy.front().at<float>(0, 2));
        }

        mean_current_queue.at<float>(0, 0) += current_queue_copy.front().at<float>(0, 0);
        mean_current_queue.at<float>(0, 1) += current_queue_copy.front().at<float>(0, 1);
        mean_current_queue.at<float>(0, 2) += current_queue_copy.front().at<float>(0, 2);

        current_queue_copy.pop();
    }
    if (VERBOSE)
        printf("\n");

    cv::Mat mean_cvMat(cv::Size(1, 3), CV_32F);

    mean_cvMat.at<float>(0, 0) = mean_current_queue.at<float>(0, 0) / (float)NUMBER_OF_FRAMES_TO_AVERAGE;
    mean_cvMat.at<float>(0, 1) = mean_current_queue.at<float>(0, 1) / (float)NUMBER_OF_FRAMES_TO_AVERAGE;
    mean_cvMat.at<float>(0, 2) = mean_current_queue.at<float>(0, 2) / (float)NUMBER_OF_FRAMES_TO_AVERAGE;

    return mean_cvMat;
}

//
// computes the mean of a std::queue<glm::vec3>. Used to average the last few camera positions to smooth trajectory
//
glm::vec3 compute_mean(std::queue<glm::vec3> previous_poses, int NUMBER_OF_FRAMES_TO_AVERAGE, bool VERBOSE = false)
{
    std::queue<glm::vec3>previous_poses_copy = previous_poses;

    glm::vec3 mean_current_pose = glm::vec3(0.0f, 0.0f, 0.0f);

    if (VERBOSE)
        printf("Queue status: ");

    while (!previous_poses_copy.empty())
    {
        if (VERBOSE)
        {
            printf("(% .02f ", previous_poses_copy.front().x);
            printf("% .02f ", previous_poses_copy.front().y);
            printf("% .02f), ", previous_poses_copy.front().z);
        }

        mean_current_pose.x += previous_poses_copy.front().x;
        mean_current_pose.y += previous_poses_copy.front().y;
        mean_current_pose.z += previous_poses_copy.front().z;

        previous_poses_copy.pop();
    }
    if (VERBOSE)
        printf("\n");

    glm::vec3 current_pos_in;

    current_pos_in.x = mean_current_pose.x / (float)NUMBER_OF_FRAMES_TO_AVERAGE;
    current_pos_in.y = mean_current_pose.y / (float)NUMBER_OF_FRAMES_TO_AVERAGE;
    current_pos_in.z = mean_current_pose.z / (float)NUMBER_OF_FRAMES_TO_AVERAGE;

    return current_pos_in;
}

// Checks if a matrix is a valid rotation matrix.
bool isRotationMatrix(cv::Mat &R)
{
    cv::Mat Rt;
    transpose(R, Rt);
    cv::Mat shouldBeIdentity = Rt * R;
    cv::Mat I = cv::Mat::eye(3, 3, shouldBeIdentity.type());

    return  cv::norm(I, shouldBeIdentity) < 1e-6;

}

int main(int argc, char **argv)
{
    ToolsC * tools = new ToolsC(BASE_PATH);
    glm::vec3 posModel2 = glm::vec3(0.0, 0.0, 0.0);
    CmdLineParser cml(argc, argv);
    // read camera parameters if passed
    if (cml["-c"])
        TheCameraParameters.readFromXMLFile(cml("-c"));

    // glfw: initialize and configure
    // ------------------------------
    glfwInit();
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

#ifdef __APPLE__
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE); // uncomment this statement to fix compilation on OS X
#endif

    // glfw window creation
    // --------------------
    GLFWwindow* window = glfwCreateWindow(SCR_WIDTH, SCR_HEIGHT, "RV – Practica tracking", NULL, NULL);
    if (window == NULL)
    {
        std::cout << "Failed to create GLFW window" << std::endl;
        glfwTerminate();
        return -1;
    }
    glfwMakeContextCurrent(window);
    glfwSetFramebufferSizeCallback(window, framebuffer_size_callback);

    // glad: load all OpenGL function pointers
    if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress))
    {
        std::cout << "Failed to initialize GLAD" << std::endl;
        return -1;
    }

    // configure global opengl state
    glEnable(GL_DEPTH_TEST);

    // build and compile shaders
    Shader ourShader(std::string(BASE_PATH + "vert.vs").c_str(), std::string(BASE_PATH + "frag.fs").c_str());
    Shader ourShader2D(std::string(BASE_PATH + "vert2D.vs").c_str(), std::string(BASE_PATH + "frag2D.fs").c_str());

    // load and create a texture 
    tools->loadTextures();

    // initializes vertex buffers
    tools->initRenderData();

    // set up shader materials
    ourShader.use();
    ourShader.setInt("material.diffuse", 0);
    ourShader.setInt("material.specular", 1);

    // set up shader materials
    ourShader2D.use();
    ourShader2D.setInt("image", 1);

    // opens video input from webcam
    TheVideoCapturer.open(0);

    // check video is open
    if (!TheVideoCapturer.isOpened())
        throw std::runtime_error("Could not open video");

    // *********************************************
    //		  -- Initialization Parameters --
    // *********************************************
    
    float TheMarkerSize = 0.04f;        // Size of markers, in meters
    float FOV = 45.0f;                  // FOV along Y in degrees
    float NearClipPlane = 0.01f;        // Near Clipping Distance
    float FarClipPlane = 1.0f;          // Far Clipping Distance

    int NUMBER_OF_FRAMES_TO_AVERAGE = 6;// Number of frames to average the last position and rotations. 
    
    int num_frames = 0;                 // Counter of frames already processed

    glm::vec3 up_vector = glm::vec3(0.0, 1.0, 0.0);                // To store the up vector of the camera
    glm::vec3 look_vector = glm::vec3(0.0, 0.0, 1.0);              // To store the look vector of the camera
    
    // render loop
    while (!glfwWindowShouldClose(window))
    {
        // increase frame count
        num_frames++;

        // to store the image from the webcam
        cv::Mat frame;

        // capture the next frame from the webcam
        TheVideoCapturer >> frame;
        cv::cvtColor(frame, frame, cv::COLOR_BGR2RGB);
        //std::cout << "Frame size: " << frame.cols << " x " << frame.rows << " x " << frame.channels() << "\n";

        if (!TheCameraParameters.isValid())
            std::cout << "ERROR: Invalid camera parameters \n";

        //***************************************************
        //              -- Inicio TAREA 1 --
        //			-- Detect and Draw MARKERS --
        //**************************************************

        TheMarkers = MDetector.detect(frame, TheCameraParameters, TheMarkerSize);
        aruco::Marker camera_pos_marker;
        for each (aruco::Marker marker in TheMarkers)
        {
            marker.draw(frame, cv::Scalar(0, 255, 0), 1);
            aruco::CvDrawingUtils::draw3dAxis(frame, marker, TheCameraParameters, 2);
            camera_pos_marker = marker;

        }

        //*************************************************** 
        //                -- Fin TAREA 1 --
        //***************************************************


        //***************************************************
        //                Inicio TAREA 2 y 3
        //		 -- Transformations and Rotations --
        //***************************************************
        cv::Mat camera_rot_matrix;
        glm::mat3 glm_rot_matrix;
        glm::mat3 glm_rot_matrix2;
        glm::mat4 new_view;
        glm::vec3 look_pos;
        glm::mat3 base_rot_matrix = glm::mat3(0);
        glm::mat3 nin_rot_matrix = glm::mat3(0);
        
        //Reflection rotation
        nin_rot_matrix[0][2] = -1.0;
        nin_rot_matrix[1][1] = -1.0;
        nin_rot_matrix[2][0] = -1.0;

        //default rotation
        base_rot_matrix[0][2] = -1.0;
        base_rot_matrix[1][1] = 1.0;
        base_rot_matrix[2][0] = 1.0;
        glm::mat3 base_inverse = glm::transpose(base_rot_matrix);
        int currentId;
        glm::mat4 model2 = glm::mat4(1.0f);
        
        if (TheMarkers.size() > 0)
        {
            for (unsigned int i = 0; i < TheMarkers.size(); i++)
            {
                currentId = TheMarkers[i].id;
                if (currentId == 177) {
                    
                     current_pos = glm::vec3(TheMarkers[i].Tvec.at<float>(0, 0), -TheMarkers[i].Tvec.at<float>(0, 1), -TheMarkers[i].Tvec.at<float>(0, 2));

                     std::cout << currentId << std::endl;
                     cv::Rodrigues(TheMarkers[i].Rvec, camera_rot_matrix);
                     fromCV2GLM_3x3(camera_rot_matrix, &glm_rot_matrix);

                     //TheMarkers[i].calculateExtrinsics(TheMarkerSize, TheCameraParameters);

                     look_vector = current_pos + glm::vec3(0.0, 0.0, 1.0);
                     look_vector = base_inverse * (nin_rot_matrix * glm_rot_matrix) * look_vector;
                 
                     up_vector = glm::vec3(0.0, 1.0, 0.0);
                     up_vector = base_inverse * (nin_rot_matrix * glm_rot_matrix) * up_vector;
                 }

                 else if (currentId == 93) {
                     posModel2 = glm::vec3(TheMarkers[i].Tvec.at<float>(0, 0), -TheMarkers[i].Tvec.at<float>(0, 1), -TheMarkers[i].Tvec.at<float>(0, 2) + 0.2);

                 }
            }
        }     

        //*************************************************** 
        //                Fin TAREA 2 y 3
        //***************************************************        
               
        
        //*************************************************** 
        //         -- Projections and Views --
        //***************************************************
        glm::mat4 current_view = glm::mat4(1.0f);
        glm::mat4 projection = glm::mat4(1.0f);

        projection = glm::perspective(FOV, (float)SCR_WIDTH / (float)SCR_HEIGHT, NearClipPlane, FarClipPlane);

        int i = 0;

        

        glm::mat4 modified_view;
        current_view = glm::lookAt(
            current_pos,    // camera pos
            look_vector,    // look at 
            up_vector		// up vector
        );
        //*************************************************** 
        //				  -- Pre-Render --
        //***************************************************
        
        // copies input image to m_textures[1]
        flip(frame, frame, -1);
        glBindTexture(GL_TEXTURE_2D, tools->m_textures[1]);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, frame.cols, frame.rows, 0, GL_RGB, GL_UNSIGNED_BYTE, frame.ptr());

        // GLFW process that reacts to input keys
        processInput(window);

        glClearColor(0.2f, 0.3f, 0.3f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT); // also clear the depth buffer now!
        while (i<2){

            float EyeSeparation = 0.0065f; // offset entre ojos
            float EyeMiddle = EyeSeparation / 2.f;
            float focus_distance = 1.0f;
            float desp_factor = atan2(EyeMiddle, focus_distance);
            
            /* *************************** */
            /* BLOQUE RENDER OJO IZQUIERDO */
            /* *************************** */
            if (i==0) {
                glColorMask(0, 1, 1, 1);
                auto leftOffset = glm::rotate(desp_factor, glm::vec3(0, 1, 0)) * glm::translate(glm::vec3(EyeMiddle, 0, 0));
                modified_view = leftOffset * current_view;
            }
            /* ************************** */
            /* BLOQUE RENDER OJO DERECHO */
            /* ************************** */
            else if (i==1) {
                glColorMask(1, 0, 0, 1);
                auto rightOffset = glm::rotate(desp_factor, glm::vec3(0, -1, 0)) * glm::translate(glm::vec3(-EyeMiddle, 0, 0));
                modified_view = rightOffset * current_view;
            }
            

            
            

            //*************************************************** 
            //				  -- RENDERING --
            //***************************************************


            // TEXTURES
            glActiveTexture(GL_TEXTURE0);
            glBindTexture(GL_TEXTURE_2D, tools->m_textures[0]);
            glActiveTexture(GL_TEXTURE1);
            glBindTexture(GL_TEXTURE_2D, tools->m_textures[1]);

            // activate shader
            ourShader.use();
            // set camera view matrix
            ourShader.setMat4("view", modified_view);

            // LIGHTS
            ourShader.setVec3("light_position", light_position);
            ourShader.setVec3("light.direction", 0.0f, 0.0f, -1.0f);
            ourShader.setVec3("light.ambient", 0.5f, 0.5f, 0.5f);
            ourShader.setVec3("light.diffuse", 0.75f, 0.75f, 0.75f);
            ourShader.setVec3("light.specular", 0.5f, 0.5f, 0.5f);

            // material properties
            ourShader.setFloat("material.shininess", 0.4f);

            ourShader.setMat4("projection", projection);

            // BOX1
            glBindVertexArray(tools->m_VAOs[2]);
            // calculate the model matrix for each object and pass it to shader before drawing
            glm::mat4 model = glm::mat4(1.0f);
            // sets model matrix
            ourShader.setMat4("model", model);

            // draws
            glDrawArrays(GL_TRIANGLES, 0, 36);
            glBindVertexArray(0);

            ourShader.setInt("material.specular", 0);

            // FLOOR
            glActiveTexture(GL_TEXTURE0);
            glBindTexture(GL_TEXTURE_2D, tools->m_textures[2]);
            glBindVertexArray(tools->m_VAOs[1]);
            ourShader.setMat4("model", model);
            glDrawArrays(GL_TRIANGLES, 0, 6);
            glBindVertexArray(0);

            // limpiamos buffer de profundidad
            glClear(GL_DEPTH_BUFFER_BIT);

            // Only last mask
            glUseProgram(0);

            //*************************************************** 
            //                TAREA 4 Opcional
            //*************************************************** 
            // TEXTURES
            glActiveTexture(GL_TEXTURE0);
            glBindTexture(GL_TEXTURE_2D, tools->m_textures[0]);
            glActiveTexture(GL_TEXTURE1);
            glBindTexture(GL_TEXTURE_2D, tools->m_textures[1]);

            // activate shader
            ourShader.use();
            // set camera view matrix
            ourShader.setMat4("view", modified_view);

            // LIGHTS
            ourShader.setVec3("light_position", light_position);
            ourShader.setVec3("light.direction", 0.0f, 0.0f, -1.0f);
            ourShader.setVec3("light.ambient", 0.5f, 0.5f, 0.5f);
            ourShader.setVec3("light.diffuse", 0.75f, 0.75f, 0.75f);
            ourShader.setVec3("light.specular", 0.5f, 0.5f, 0.5f);

            // material properties
            ourShader.setFloat("material.shininess", 0.4f);

            ourShader.setMat4("projection", projection);

            
            // BOX2
            glBindVertexArray(tools->m_VAOs[2]);
            // calculate the model matrix for each object and pass it to shader before drawing
            
            model2[3][0] = posModel2[0];
            model2[3][1] = posModel2[1];
            model2[3][2] = posModel2[2];
            // sets model matrix
            ourShader.setMat4("model", model2);

            // draws
            glDrawArrays(GL_TRIANGLES, 0, 36);
            glBindVertexArray(0);

            ourShader.setInt("material.specular", 0);

            // limpiamos buffer de profundidad
            glClear(GL_DEPTH_BUFFER_BIT);

            // Only last mask
            glUseProgram(0);

            i++;
        }

        // *************************** 
        //         - WEBCAM -
        // ***************************

        // configures projection matrix
        projection = glm::perspective(glm::radians(45.0f), (float)SCR_WIDTH / (float)SCR_HEIGHT, 0.1f, 100.0f);

        glColorMask(true, true, true, false);
        glActiveTexture(GL_TEXTURE1);
        glBindTexture(GL_TEXTURE_2D, tools->m_textures[1]);

        ourShader2D.use();
        float screen_width = 2.0f;
        glm::mat4 projection2D = glm::ortho(0.0f, (float)SCR_WIDTH, 0.0f, (float)SCR_HEIGHT, -1.0f, 1.0f);
        glm::mat4 model2D = glm::translate(glm::mat4(1.0f), glm::vec3((float)SCR_WIDTH / 2.0f - 160.0f, 540.0f, 0.f));

        ourShader2D.setMat4("projection2D", projection2D);
        //std::cout << "glGetUniformLocation " << " :" << glGetUniformLocation(ourShader2D.ID, "projection2D");
        ourShader2D.setMat4("model2D", model2D);

        glBindVertexArray(tools->m_VAOs[0]);                 //VAOs[0] is 2D quad for cam input
        glDrawArrays(GL_TRIANGLES, 0, 6);
        glBindVertexArray(0);

        glUseProgram(0);

        // glfw: swap buffers and poll IO events (keys pressed/released, mouse moved etc.)
        glfwSwapBuffers(window);
        glfwPollEvents();
        
    }

    // de-allocate all resources once they've outlived their purpose:
    glDeleteVertexArrays(1, &(tools->m_VAOs[2]));
    glDeleteBuffers(1, &(tools->m_VBOs[2]));

    glDeleteVertexArrays(1, &(tools->m_VAOs[1]));
    glDeleteBuffers(1, &(tools->m_VBOs[1]));

    glDeleteVertexArrays(1, &(tools->m_VAOs[0]));
    glDeleteBuffers(1, &(tools->m_VBOs[0]));

    // glfw: terminate, clearing all previously allocated GLFW resources.
    glfwTerminate();

    std::cout << "Bye!" << std::endl;
    return 0;
}

// process all input: query GLFW whether relevant keys are pressed/released this frame and react accordingly
void processInput(GLFWwindow *window)
{
    if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS)
        glfwSetWindowShouldClose(window, true);
}

// glfw: whenever the window size changed (by OS or user resize) this callback function executes
void framebuffer_size_callback(GLFWwindow* window, int width, int height)
{
    // make sure the viewport matches the new window dimensions; note that width and 
    // height will be significantly larger than specified on retina displays.
    glViewport(0, 0, width, height);
}