# OpenCV
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

# Eigen
# find_package (Eigen3 3.3 REQUIRED)

# OpenMP
find_package(OpenMP REQUIRED)

# boost::timer
find_package(Boost 1.48 REQUIRED COMPONENTS timer)

include_directories(../../external/pmp-library/external/eigen ../../external/nvdiffrast)

file(GLOB PIMP_CXX "*.cpp" "algorithms/*.cpp")

add_library(smp ${PIMP_CXX})
target_link_libraries(smp PUBLIC nvdiffrast pmp "${TORCH_LIBRARIES}" "${Python_LIBRARIES}" ${Boost_LIBRARIES} ${Boost_TIMER_LIBRARY})

# target_link_libraries(smp OpenMP::OpenMP_CXX nvdiffrast pmp)
# target_include_directories(smp PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})