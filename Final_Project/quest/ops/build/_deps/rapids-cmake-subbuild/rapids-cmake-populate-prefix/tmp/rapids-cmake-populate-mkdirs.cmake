# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file Copyright.txt or https://cmake.org/licensing for details.

cmake_minimum_required(VERSION ${CMAKE_VERSION}) # this file comes with cmake

# If CMAKE_DISABLE_SOURCE_CHANGES is set to true and the source directory is an
# existing directory in our source tree, calling file(MAKE_DIRECTORY) on it
# would cause a fatal error, even though it would be a no-op.
if(NOT EXISTS "/home/fanz2/quest/quest/ops/build/_deps/rapids-cmake-src")
  file(MAKE_DIRECTORY "/home/fanz2/quest/quest/ops/build/_deps/rapids-cmake-src")
endif()
file(MAKE_DIRECTORY
  "/home/fanz2/quest/quest/ops/build/_deps/rapids-cmake-build"
  "/home/fanz2/quest/quest/ops/build/_deps/rapids-cmake-subbuild/rapids-cmake-populate-prefix"
  "/home/fanz2/quest/quest/ops/build/_deps/rapids-cmake-subbuild/rapids-cmake-populate-prefix/tmp"
  "/home/fanz2/quest/quest/ops/build/_deps/rapids-cmake-subbuild/rapids-cmake-populate-prefix/src/rapids-cmake-populate-stamp"
  "/home/fanz2/quest/quest/ops/build/_deps/rapids-cmake-subbuild/rapids-cmake-populate-prefix/src"
  "/home/fanz2/quest/quest/ops/build/_deps/rapids-cmake-subbuild/rapids-cmake-populate-prefix/src/rapids-cmake-populate-stamp"
)

set(configSubDirs )
foreach(subDir IN LISTS configSubDirs)
    file(MAKE_DIRECTORY "/home/fanz2/quest/quest/ops/build/_deps/rapids-cmake-subbuild/rapids-cmake-populate-prefix/src/rapids-cmake-populate-stamp/${subDir}")
endforeach()
if(cfgdir)
  file(MAKE_DIRECTORY "/home/fanz2/quest/quest/ops/build/_deps/rapids-cmake-subbuild/rapids-cmake-populate-prefix/src/rapids-cmake-populate-stamp${cfgdir}") # cfgdir has leading slash
endif()
