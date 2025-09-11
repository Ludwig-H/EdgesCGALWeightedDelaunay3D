cmake_minimum_required(VERSION 3.22)
project(EdgesCGALWeightedDelaunay3D LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

# Silence Boost CMP0167 dev warning in newer CMake without changing user policies
if(POLICY CMP0167)
  cmake_policy(SET CMP0167 NEW)
endif()

find_package(CGAL REQUIRED)
# Try TBB; enable parallel mode only if we also find tbbmalloc
find_package(TBB QUIET)

set(USE_TBB OFF)
set(TBB_MALLOC_TARGET "")
if(TBB_FOUND)
  if(TARGET TBB::tbb)
    set(USE_TBB ON)
  endif()
  if(TARGET TBB::tbbmalloc)
    set(TBB_MALLOC_TARGET TBB::tbbmalloc)
  else()
    # Fallback: search the lib directly (Debian/Ubuntu often split it)
    find_library(TBBMALLOC_LIB tbbmalloc)
    if(TBBMALLOC_LIB)
      set(TBB_MALLOC_TARGET ${TBBMALLOC_LIB})
    endif()
  endif()
  if(USE_TBB AND TBB_MALLOC_TARGET STREQUAL "")
    message(WARNING "TBB found but tbbmalloc missing => disabling parallel CGAL path.")
    set(USE_TBB OFF)
  endif()
endif()

add_executable(EdgesCGALWeightedDelaunay3D src/EdgesCGALWeightedDelaunay3D.cpp)
target_link_libraries(EdgesCGALWeightedDelaunay3D PRIVATE CGAL::CGAL)

if(USE_TBB)
  target_link_libraries(EdgesCGALWeightedDelaunay3D PRIVATE TBB::tbb ${TBB_MALLOC_TARGET})
  target_compile_definitions(EdgesCGALWeightedDelaunay3D PRIVATE CGAL_LINKED_WITH_TBB)
endif()

# Speed knobs
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  target_compile_options(EdgesCGALWeightedDelaunay3D PRIVATE -O3 -DNDEBUG -march=native)
elseif(MSVC)
  target_compile_options(EdgesCGALWeightedDelaunay3D PRIVATE /O2 /DNDEBUG)
endif()
