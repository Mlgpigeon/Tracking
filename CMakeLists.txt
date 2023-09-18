# This CMakeLists is intended only for macOS and Linux.

cmake_minimum_required(VERSION 2.8)

# Set a project name.
project(rv_practica_tracking_v1)

# Use the C++11 standard.
set(CMAKE_INCLUDE_CURRENT_DIR ON)
set(CMAKE_CXX_STANDARD 11) # C++11...
set(CMAKE_CXX_STANDARD_REQUIRED ON) #...is required...
set(CMAKE_CXX_EXTENSIONS ON) #...with compiler extensions like gnu++11

# Suppress warnings of the deprecation of glut functions on macOS.
if(APPLE)
   add_definitions(-Wno-deprecated-declarations)
endif()

if(WIN32)
   # for Windows operating system in general
   # set(OpenCV_DIR "3rdparty/opencv")
   # message(STATUS "OpenCV found, version: ${OpenCV_VERSION} in dir ${OpenCV_INCLUDE_DIRS}")
   # INCLUDE_DIRECTORIES("3rdparty/opencv/include")

   find_package(OpenCV REQUIRED)
   include_directories( ${OpenCV_INCLUDE_DIRS} )
   message(STATUS "OpenCV found, version: ${OpenCV_VERSION} in dir ${OpenCV_INCLUDE_DIRS}")

   set(INCLUDE_DIRS "3rdparty/freeglut/include")
   link_directories(${CMAKE_SOURCE_DIR}/3rdparty/opencv/x86/vc15/lib)
   link_directories(${CMAKE_SOURCE_DIR}/3rdparty/opencv/x86/vc15/bin)

   find_package(OpenGL REQUIRED)
   message(STATUS "OpenGL found in dir ${OPENGL_INCLUDE_DIR}")

   link_directories(${CMAKE_SOURCE_DIR}/3rdparty/freeglut/lib)
endif()

if(APPLE)
   find_package(OpenGL REQUIRED)
   find_package(GLUT REQUIRED)
   set(INCLUDE_DIRS ${OPENGL_INCLUDE_DIR} ${GLUT_INCLUDE_DIR})
endif()

# OPENGL_INCLUDE_DIR, GLUT_INCLUDE_DIR, OPENGL_LIBRARIES, and GLUT_LIBRARIES
# are CMake built-in variables defined when the packages are found.
set(LIBRARIES ${OPENGL_LIBRARIES} ${GLUT_LIBRARIES})

# If not on macOS, add glew include directory and library path to lists.
if(UNIX AND NOT APPLE) 
   list(APPEND INCLUDE_DIRS ${GLEW_INCLUDE_DIRS})
   list(APPEND LIBRARIES ${GLEW_LIBRARIES})
endif()

message(STATUS "OpenGL found in dir ${OPENGL_INCLUDE_DIR}")

# Add the list of include paths to be used to search for include files.
include_directories(${INCLUDE_DIRS})

INCLUDE_DIRECTORIES(${CMAKE_SOURCE_DIR}/3rdparty/aruco/include/src)
link_directories(${CMAKE_SOURCE_DIR}/3rdparty/aruco/lib)

if(APPLE)
   find_package(OpenCV REQUIRED)
   include_directories(${OpenCV_INCLUDE_DIRS} )
   message(STATUS "OpenCV found, version: ${OpenCV_VERSION} in dir ${OpenCV_INCLUDE_DIRS}")
endif()

# GLFW
add_subdirectory(3rdparty/glfw-3.3.2/)
include_directories(3rdparty/glfw-3.3.2/include)

# glad
add_subdirectory(3rdparty/glad/)

# glm
INCLUDE_DIRECTORIES(${CMAKE_SOURCE_DIR}/3rdparty/glm/include)

list(APPEND LIBRARIES opencv_core opencv_calib3d opencv_imgcodecs opencv_videoio opencv_imgproc opencv_features2d opencv_highgui ${THREADLIB})
list(APPEND LIBRARIES aruco) 
list(APPEND LIBRARIES glfw ${GLFW_LIBRARIES})
list(APPEND LIBRARIES glad)

if(CMAKE_COMPILER_IS_GNUCXX OR MINGW OR ${CMAKE_CXX_COMPILER_ID} STREQUAL Clang)
SET(THREADLIB "pthread")
ENDIF()

# Search all the .cpp files in the directory where CMakeLists lies and set them to ${SOURCE_FILES}.
# Search all the .h files in the directory where CMakeLists lies and set them to ${INCLUDE_FILES}.
file(GLOB SOURCE_FILES ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp)
file(GLOB INCLUDE_FILES ${CMAKE_CURRENT_SOURCE_DIR}/*.h)

# Add the executable Example to be built from the source files.
add_executable(rv_practica_tracking_v1 ${SOURCE_FILES} ${SOURCE_FILES_TOOLS} ${INCLUDE_FILES})

# Link the executable to the libraries.
target_link_libraries(rv_practica_tracking_v1 ${LIBRARIES})

set( CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin )
set( CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib )

if(WIN32)
   add_custom_command(
      TARGET rv_practica_tracking_v1 
      POST_BUILD
      COMMAND ${CMAKE_COMMAND} -E copy
         ${OpenCV_DIR}/bin/Release/opencv_videoio450.dll
         $<TARGET_FILE_DIR:rv_practica_tracking_v1>
   )

   add_custom_command(
      TARGET rv_practica_tracking_v1 
      POST_BUILD
      COMMAND ${CMAKE_COMMAND} -E copy
         ${OpenCV_DIR}/bin/Release/opencv_core450.dll
         $<TARGET_FILE_DIR:rv_practica_tracking_v1>
   )

   add_custom_command(
      TARGET rv_practica_tracking_v1 
      POST_BUILD
      COMMAND ${CMAKE_COMMAND} -E copy
         ${OpenCV_DIR}/bin/Release/opencv_imgproc450.dll
         $<TARGET_FILE_DIR:rv_practica_tracking_v1>
   )

   add_custom_command(
      TARGET rv_practica_tracking_v1 
      POST_BUILD
      COMMAND ${CMAKE_COMMAND} -E copy
         ${OpenCV_DIR}/bin/Release/opencv_features2d450.dll
         $<TARGET_FILE_DIR:rv_practica_tracking_v1>
   )

   add_custom_command(
      TARGET rv_practica_tracking_v1 
      POST_BUILD
      COMMAND ${CMAKE_COMMAND} -E copy
         ${OpenCV_DIR}/bin/Release/opencv_flann450.dll
         $<TARGET_FILE_DIR:rv_practica_tracking_v1>
   )

   add_custom_command(
      TARGET rv_practica_tracking_v1 
      POST_BUILD
      COMMAND ${CMAKE_COMMAND} -E copy
         ${OpenCV_DIR}/bin/Release/opencv_calib3d450.dll
         $<TARGET_FILE_DIR:rv_practica_tracking_v1>
   )

   add_custom_command(
      TARGET rv_practica_tracking_v1 
      POST_BUILD
      COMMAND ${CMAKE_COMMAND} -E copy
         ${OpenCV_DIR}/bin/Release/opencv_imgcodecs450.dll
         $<TARGET_FILE_DIR:rv_practica_tracking_v1>
   )

   add_custom_command(
      TARGET rv_practica_tracking_v1 
      POST_BUILD
      COMMAND ${CMAKE_COMMAND} -E copy
         ${CMAKE_SOURCE_DIR}/3rdparty/aruco/lib/aruco3112.dll
         $<TARGET_FILE_DIR:rv_practica_tracking_v1>
   )

   add_custom_command(
      TARGET rv_practica_tracking_v1 
      POST_BUILD
      COMMAND ${CMAKE_COMMAND} -E copy
         ${CMAKE_SOURCE_DIR}/3rdparty/freeglut/bin/freeglut.dll
         $<TARGET_FILE_DIR:rv_practica_tracking_v1>
   )
endif()