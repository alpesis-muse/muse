CMAKE_MINIMUM_REQUIRED ( VERSION 2.8 )

# ----------------------------------------------------------------------------
# Project settings

# project name
PROJECT ( MUSE CXX C )

# project version
SET ( MUSE_MAJOR_VERSION 0 )
SET ( MUSE_MINOR_VERSION 0 )
SET ( MUSE_PATCH_VERSION 1 )
SET ( MUSE_VERSION ${MUSE_MAJOR_VERSION}.${MUSE_MINOR_VERSION}.${MUSE_PATCH_VERSION} )

# ----------------------------------------------------------------------------
# CMake settings

# cmake packages
SET ( CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake" )

# libraries
INCLUDE ( enableCXX11 )

# ----------------------------------------------------------------------------
# Option settings


# ----------------------------------------------------------------------------
# Sources settings

SET ( PROJECT_ROOT ${CMAKE_SOURCE_DIR} )
INCLUDE_DIRECTORIES ( "${PROJECT_ROOT}" )

# ----------------------------------------------------------------------------
# Subdirectories

ADD_SUBDIRECTORY ( muse )
