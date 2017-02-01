#ifndef MATHART_H_
#define MATHART_H_

#include <cmath>
#include <iostream>


namespace muse
{
// -----------------------------------------------------------------------------------------------

#define DIM 1024
#define DM1 (DIM-1)

#define _sq(x) ((x)*(x))                           // square
#define _cb(x) abs((x)*()*(x))                     // absolute value of cube 
#define _cr(x) (unsigned short)(pow((x),1.0/3.0))  // cube root

unsigned short RED (int i, int j);
unsigned short GREEN (int i, int j);
unsigned short BLUE (int i, int j);

void writePixel (int i, int j, FILE *fp);
void generateImage ();

// -----------------------------------------------------------------------------------------------
}  // namespace: muse

#endif  // MATHART_H_
