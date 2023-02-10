
# Rete
add_executable(rete rete.cpp)

target_link_libraries(rete
        features
        ${llvm_libs}
        clangTooling
        clangBasic
        clangASTMatchers
        )
set_target_properties(rete PROPERTIES COMPILE_FLAGS "-fno-rtti -fno-exceptions" ) # this is to be compatible with llvm libraries

