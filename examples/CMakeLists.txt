include_directories(${CMAKE_CURRENT_BINARY_DIR})
include_directories(${CMAKE_CURRENT_SOURCE_DIR})

add_executable( DamBreak DamBreak.cpp )
target_link_libraries( DamBreak PRIVATE exaclamr)
target_include_directories( DamBreak PRIVATE ${CMAKE_CURRENT_BINARY_DIR})

#add_executable( TestHilbert TestHilbert.cpp )
#target_link_libraries( TestHilbert PRIVATE exaclamr )
#target_include_directories( TestHilbert PRIVATE ${CMAKE_CURRENT_BINARY_DIR} )

add_executable( 3DHalo 3DHalo.cpp )
target_link_libraries( 3DHalo PRIVATE exaclamr )
target_include_directories( 3DHalo PRIVATE ${CMAKE_CURRENT_BINAY_IR} )
