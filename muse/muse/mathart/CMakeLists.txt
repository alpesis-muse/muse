# package

# ----------------------------------------------------------------------------
# mathart

FILE ( GLOB MATHART_HEADERS . *.h )
FILE ( GLOB MATHART_SOURCES . *.cpp )
ADD_LIBRARY ( muse_mathart STATIC ${MATHART_SOURCES} )

