# laundre

Clone inside homework folder (cs225a/homework)

Edit CMakeLists.txt inside homework folder to:

set(HW_FOLDER "${CMAKE_CURRENT_SOURCE_DIR}")
add_definitions(-DHW_FOLDER="${HW_FOLDER}")

add_subdirectory(hw0)
add_subdirectory(hw1)
add_subdirectory(hw2)
add_subdirectory(hw3)
add_subdirectory(laundre)
