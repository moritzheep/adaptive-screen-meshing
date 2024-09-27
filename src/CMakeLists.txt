add_subdirectory(smp)

#Python
find_package(Python COMPONENTS Development REQUIRED)
include_directories(${Python_INCLUDE_DIRS})

# OpenCV
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

include_directories(../external/pmp-library/external/eigen ../external/nvdiffrast)
include_directories(../external/pmp-library/src)

add_executable(main main.cpp)
target_link_libraries(main PUBLIC pmp nvdiffrast ${OpenCV_LIBS} smp "${TORCH_LIBRARIES}" "${Python_LIBRARIES}")
