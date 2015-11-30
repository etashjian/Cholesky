cmake_minimum_required(VERSION 2.8)
project(Cholesky)

#The following will but all binary files in a "bin" directory
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

#increase warning levels
set(CMAKE_C_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Werror")

# add include path
include_directories(.)

add_subdirectory(test)
