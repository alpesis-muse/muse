# package:

# ----------------------------------------------------------------------------
# Subdirectories

ADD_SUBDIRECTORY ( mathart )

# ----------------------------------------------------------------------------
# main


FILE ( GLOB MUSE_HEADERS . *.h )
FILE ( GLOB MUSE_SOURCES . *.cpp )

ADD_EXECUTABLE ( muse ${MUSE_SOURCES} )
TARGET_LINK_LIBRARIES ( muse muse_mathart )
