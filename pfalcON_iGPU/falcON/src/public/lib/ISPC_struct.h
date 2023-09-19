/*
 * Copyright (c) 2013       Benoit Lange, Pierre Fortin
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program; if not, write to the Free Software
 * Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA
 *
 */
////////////////////////////////////////////////////////////////////////////////
//          22/11/2013  BL recursive implementation of DTT
//          22/11/2013  BL added support of OpenMP for parallel DTT
//          22/11/2013  BL added support of TBB for parallel DTT
// v p0.1   22/11/2013  BL added ISPC kernels
////////////////////////////////////////////////////////////////////////////////

#if withoutNR
#define rsqrtNR(v) v= rsqrt(v);
#else
#define rsqrtNR(v) { varying float temp = rsqrt(v); temp *= (temp * temp * (v) - 3.0f) * (-0.5f); v = temp; }
#endif


struct pfalcONstruct
{
    unsigned int id;
    float SCAL;
    
    float a[3];
    
    int padding[7];
    
};

struct pfalcONstructPROP
{
    float POT[4];
};
