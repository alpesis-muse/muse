#include "muse/mathart/mathart.h"


namespace muse
{
// -----------------------------------------------------------------------------------------------


unsigned short RED (int i, int j)
{

}


unsigned short GREEN (int i, int j)
{

}


unsigned short BLUE (int i, int j)
{

}


unsigned short setColor (int i, int j)
{
    unsigned short color[3];
    color[0] = RED(i,j)&DM1;
    color[1] = GREEN(i,j)&DM1;
    color[2] = BLUE(i,j)&DM1;

    return *color;
}


void writePixel (unsigned short *color, FILE *fp)
{
    fwrite (color, 2, 3, fp);
}


void generateImage ()
{
    FILE *fp;
    fp = fopen ("MathematicalImage", "wb");
    fprintf(fp, "P6\n%d %d\n1023\n", DIM, DIM);

    for (int j = 0; j < DIM; ++j)
    {
        for (int i = 0; i < DIM; ++i)
        {
            unsigned short color = setColor (i, j);
            writePixel (&color, fp);
        }
    }

    fclose (fp);
}

// -----------------------------------------------------------------------------------------------
}  // namespace: muse
