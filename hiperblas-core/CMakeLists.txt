cmake_minimum_required(VERSION 3.20)
project(hiperblas-core LANGUAGES C CXX)

# Ativar testes
include(CTest)
enable_testing()

# Configuração do C padrão
set(CMAKE_C_STANDARD 99)
set(CMAKE_C_STANDARD_REQUIRED ON)

# Compilação mais restrita e debug friendly
add_compile_options(       -Wextra -Wpedantic -fexceptions -O3 -g3 -ggdb)
#add_compile_options( -Wall -Wextra -Wpedantic -fexceptions -O3 -g3 -ggdb)

# Threads
set(THREADS_PREFER_PTHREAD_FLAG ON)
find_package(Threads REQUIRED)

# OpenMP (opcional)
find_package(OpenMP)
if(OPENMP_FOUND)
    set(CMAKE_C_FLAGS   "${CMAKE_C_FLAGS}   ${OpenMP_C_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()

# Caminhos importantes
set(HEADER_PATH       ${PROJECT_SOURCE_DIR}/include)
set(SOURCE_PATH       ${PROJECT_SOURCE_DIR}/src)
set(DIST_PATH         ${PROJECT_SOURCE_DIR}/dist)
set(DIST_INCLUDE_PATH ${DIST_PATH}/include)
set(PROJECT_TEST_DIR  ${PROJECT_SOURCE_DIR}/test)


include_directories(${HEADER_PATH})

# Bibliotecas principais
add_library(hiperblas-core SHARED
    ${SOURCE_PATH}/libhiperblas.c
    ${SOURCE_PATH}/hiperblas_std.c
)

add_library(hiperblas-cpu-bridge SHARED
    ${SOURCE_PATH}/libhiperblas-cpu-bridge-vector.c
    ${SOURCE_PATH}/libhiperblas-cpu-bridge-list.c
    ${SOURCE_PATH}/libhiperblas-cpu-bridge-matrix.c
    ${SOURCE_PATH}/libhiperblas-cpu-bridge-smatrix.c
    ${SOURCE_PATH}/libhiperblas-cpu-bridge-complex.c
    ${SOURCE_PATH}/libhiperblas-cpu-bridge-api-impl.c
)

target_link_libraries(hiperblas-core PRIVATE Threads::Threads ${CMAKE_DL_LIBS})

# === Testes ===
find_package(GTest REQUIRED)

# === Testes ===
# Tenta achar o GTest, mas também permite o usuário passar manualmente
#BDfind_package(GTest)
find_package(GTest REQUIRED)
include_directories(${GTEST_INCLUDE_DIRS})

#add_executable(vector_test test/vector_test.cpp)


#BD if (NOT GTest_FOUND)
#BD     message(STATUS "GTest não encontrado automaticamente, usando valores manuais")
#BD 
    # Caminho para includes
    #BD     set(GTEST_INCLUDE_DIR "/usr/local/include" CACHE PATH "Path to GTest includes")
    #BD include_directories(${GTEST_INCLUDE_DIR})

    # Caminho para libs
    #BD set(GTEST_LIBRARY "/usr/local/lib/libgtest.a" CACHE FILEPATH "Path to gtest library")
    #BD set(GTEST_MAIN_LIBRARY "/usr/local/lib/libgtest_main.a" CACHE FILEPATH "Path to gtest_main library")

    #BD set(GTEST_BOTH_LIBRARIES ${GTEST_LIBRARY} ${GTEST_MAIN_LIBRARY})
    #BD else()
	#BD include_directories(${GTEST_INCLUDE_DIRS})
	#BD set(GTEST_BOTH_LIBRARIES ${GTEST_LIBRARIES})
	#BD endif()


#add_executable(vector_test ${PROJECT_TEST_DIR}/vector_test.cpp)
#target_link_libraries(vector_test PRIVATE hiperblas-core ${GTEST_BOTH_LIBRARIES} Threads::Threads)
#BD target_link_libraries(vector_test PRIVATE ${GTEST_LIBRARIES} pthread)

#add_executable(matrix_test ${PROJECT_TEST_DIR}/matrix_test.cpp)
#target_link_libraries(matrix_test PRIVATE hiperblas-core ${GTEST_BOTH_LIBRARIES} Threads::Threads)

add_executable(sparse_matrix_test ${PROJECT_TEST_DIR}/sparse_matrix_test.cpp)
target_link_libraries(sparse_matrix_test PRIVATE hiperblas-core ${GTEST_BOTH_LIBRARIES} Threads::Threads)

#add_executable(hiperblas_test ${PROJECT_TEST_DIR}/hiperblas_test.cpp)
#target_link_libraries(hiperblas_test PRIVATE hiperblas-core ${GTEST_BOTH_LIBRARIES} Threads::Threads)

#add_test(NAME vector_test        COMMAND vector_test)
#add_test(NAME matrix_test        COMMAND matrix_test)
add_test(NAME sparse_matrix_test COMMAND sparse_matrix_test)
#add_test(NAME hiperblas_test       COMMAND hiperblas_test)

# === Instalação ===
include(GNUInstallDirs)

install(TARGETS hiperblas-core DESTINATION ${CMAKE_INSTALL_LIBDIR})
install(TARGETS hiperblas-cpu-bridge DESTINATION ${CMAKE_INSTALL_LIBDIR})

install(FILES
    ${HEADER_PATH}/hiperblas.h
    ${HEADER_PATH}/hiperblas_list.h
    ${HEADER_PATH}/hiperblas_std.h
    ${HEADER_PATH}/hiperblas_vector.h
    ${HEADER_PATH}/hiperblas_matrix.h
    ${HEADER_PATH}/hiperblas_smatrix.h
    ${HEADER_PATH}/hiperblas_complex.h
    ${HEADER_PATH}/libhiperblas.h
    ${HEADER_PATH}/bridge_api.h
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

# === Alvos auxiliares ===

# Código cobertura
option(CODE_COVERAGE "Build with code coverage" OFF)
if(CODE_COVERAGE AND CMAKE_COMPILER_IS_GNUCXX)
    message(STATUS "Building with lcov Code Coverage Tools")
    set(CMAKE_C_FLAGS   "${CMAKE_C_FLAGS}   --coverage -fprofile-arcs -ftest-coverage")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} --coverage -fprofile-arcs -ftest-coverage")

    add_custom_target(coverage
        COMMAND lcov --directory . --capture --output-file coverage.info
        COMMAND lcov --remove coverage.info '/usr/*' --output-file coverage.info
        COMMAND lcov --list coverage.info
        WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
        COMMENT "Generating code coverage report"
    )

    add_custom_target(coverage-html
        COMMAND genhtml coverage.info --output-directory coverage-html
        WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
        COMMENT "Generating HTML code coverage report"
    )
endif()

