#include "muse/mathart/mathart.h"


namespace muse
{
// -----------------------------------------------------------------------------------------------


unsigned short RED (int i, int j)
{

   double a=0,b=0,c,d,n=0;
   while((c=a*a)+(d=b*b)<4&&n++<880)
   {b=2*a*b+j*8e-9-.645411;a=c-d+i*8e-9+.356888;}

   return 255*pow((n-80)/800,3.);
}


unsigned short GREEN (int i, int j)
{
   double a=0,b=0,c,d,n=0;
   while((c=a*a)+(d=b*b)<4&&n++<880)
   {b=2*a*b+j*8e-9-.645411;a=c-d+i*8e-9+.356888;}

   return 255*pow((n-80)/800,.7);
}


unsigned short BLUE (int i, int j)
{
   double a=0,b=0,c,d,n=0;
   while((c=a*a)+(d=b*b)<4&&n++<880)
   {b=2*a*b+j*8e-9-.645411;a=c-d+i*8e-9+.356888;}

   return 255*pow((n-80)/800,.5);
}



void writePixel (int i, int j, FILE *fp)
{
    unsigned short color[3];
    color[0] = RED(i,j)&DM1;
    color[1] = GREEN(i,j)&DM1;
    color[2] = BLUE(i,j)&DM1;
    fwrite (color, 2, 3, fp);
}


void generateImage ()
{
    FILE *fp;
    fp = fopen ("MathematicalImage.jpg", "wb");
    fprintf(fp, "P6\n%d %d\n1023\n", DIM, DIM);

    for (int j = 0; j < DIM; ++j)
    {
        for (int i = 0; i < DIM; ++i)
        {
            writePixel (i, j, fp);
        }
    }

    fclose (fp);
}

// -----------------------------------------------------------------------------------------------
}  // namespace: muse
