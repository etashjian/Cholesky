FILE(GLOB db ../db/*)
add_executable(simple simple.cpp ${db})
add_executable(simple_block simple_block.cpp ${db})
add_executable(simple_band simple_band.cpp ${db})
