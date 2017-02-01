# package:

# ----------------------------------------------------------------------------

FILE ( GLOB MUSE_HEADERS . *.h )
FILE ( GLOB MUSE_SOURCES . *.cpp )

ADD_EXECUTABLE ( muse ${MUSE_SOURCES} )
