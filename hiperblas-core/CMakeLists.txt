cmake_minimum_required(VERSION 3.20)

project(hiperblas
    VERSION 0.4
    LANGUAGES C CXX
)

# =====================================================
# Opções globais
# =====================================================

include(GNUInstallDirs)
include(CTest)

set(CMAKE_C_STANDARD 99)
set(CMAKE_C_STANDARD_REQUIRED ON)

set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Flags comuns (sem OpenMP aqui!)
add_compile_options(
    -Wall
    -Wextra
    -Wpedantic
    -O3
    -g3
    -ggdb
)

# =====================================================
# RPATH – elimina LD_LIBRARY_PATH
# =====================================================

set(CMAKE_SKIP_BUILD_RPATH FALSE)
set(CMAKE_BUILD_WITH_INSTALL_RPATH TRUE)
set(CMAKE_INSTALL_RPATH "$ORIGIN")
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)

# =====================================================
# Dependências
# =====================================================

find_package(Threads REQUIRED)
find_package(OpenMP REQUIRED)

# GoogleTest é opcional (só para testes)
find_package(GTest QUIET)

# =====================================================
# Paths
# =====================================================

set(SRC_DIR  ${PROJECT_SOURCE_DIR}/src)
set(INC_DIR  ${PROJECT_SOURCE_DIR}/include)
set(TEST_DIR ${PROJECT_SOURCE_DIR}/test)

include_directories(${INC_DIR})

# =====================================================
# hiperblas-core
# =====================================================

add_library(hiperblas-core SHARED
    ${SRC_DIR}/libhiperblas.c
    ${SRC_DIR}/hiperblas_std.c
)

target_link_libraries(hiperblas-core
    PRIVATE
        Threads::Threads
        OpenMP::OpenMP_C
        ${CMAKE_DL_LIBS}
)

set_target_properties(hiperblas-core PROPERTIES
    OUTPUT_NAME hiperblas-core
    INSTALL_RPATH "$ORIGIN"
)

# =====================================================
# hiperblas-cpu-bridge
# =====================================================

add_library(hiperblas-cpu-bridge SHARED
    ${SRC_DIR}/libhiperblas-cpu-bridge-api-impl.c
    ${SRC_DIR}/libhiperblas-cpu-bridge-list.c
    ${SRC_DIR}/libhiperblas-cpu-bridge-vector.c
    ${SRC_DIR}/libhiperblas-cpu-bridge-matrix.c
    ${SRC_DIR}/libhiperblas-cpu-bridge-smatrix.c
    ${SRC_DIR}/libhiperblas-cpu-bridge-complex.c
)

target_link_libraries(hiperblas-cpu-bridge
    PRIVATE
        hiperblas-core
        Threads::Threads
        OpenMP::OpenMP_C
)

set_target_properties(hiperblas-cpu-bridge PROPERTIES
    OUTPUT_NAME hiperblas-cpu-bridge
    INSTALL_RPATH "$ORIGIN"
)

# =====================================================
# Testes (GoogleTest – opcionais)
# =====================================================

if(GTest_FOUND)
    enable_testing()

    set(HIPERBLAS_TESTS
        sparseMatrixTest
        vectorTest
    )

    foreach(test_name IN LISTS HIPERBLAS_TESTS)

        add_executable(${test_name}
            ${TEST_DIR}/${test_name}.cpp
        )

        target_link_libraries(${test_name}
            PRIVATE
                hiperblas-core
                GTest::gtest
                GTest::gtest_main
                Threads::Threads
                OpenMP::OpenMP_C
        )

        add_test(NAME ${test_name} COMMAND ${test_name})

    endforeach()
endif()

# =====================================================
# Instalação
# =====================================================

install(TARGETS
    hiperblas-core
    hiperblas-cpu-bridge
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
)

install(FILES
    ${INC_DIR}/hiperblas.h
    ${INC_DIR}/hiperblas_std.h
    ${INC_DIR}/hiperblas_list.h
    ${INC_DIR}/hiperblas_vector.h
    ${INC_DIR}/hiperblas_matrix.h
    ${INC_DIR}/hiperblas_smatrix.h
    ${INC_DIR}/hiperblas_complex.h
    ${INC_DIR}/libhiperblas.h
    ${INC_DIR}/bridge_api.h
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

