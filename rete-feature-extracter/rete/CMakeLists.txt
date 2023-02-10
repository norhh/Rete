
add_library (features
            feature_matchers.cpp
            feature_vector.cpp
            rete_utils.cpp
        )

target_include_directories (features PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

set_target_properties(features PROPERTIES COMPILE_FLAGS "-fno-rtti -fno-exceptions" ) # this is to be compatible with llvm libraries