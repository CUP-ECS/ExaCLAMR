include_directories(${CMAKE_CURRENT_BINARY_DIR})
include_directories(${CMAKE_CURRENT_SOURCE_DIR})

add_executable( DamBreak dam_break.cpp )
target_link_libraries( DamBreak PRIVATE exaclamr)
target_include_directories( DamBreak PRIVATE ${CMAKE_CURRENT_BINARY_DIR})
