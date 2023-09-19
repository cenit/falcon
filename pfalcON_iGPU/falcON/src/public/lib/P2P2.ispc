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

#include "ISPC_struct.h"

export void cellLeaf2(uniform pfalcONstruct A[], uniform pfalcONstruct B[], uniform int ni, uniform float EQ, uniform pfalcONstructPROP ACPN[])
{
  foreach (i = 0 ... ni)
    {
      uniform int id1 = 0;
      uniform int j = 0;
        
      varying float m0 =  (i < ni) ? A[i].SCAL : 0.0 ;
        
      varying float xi = A[i].a[0];
      varying float yi = A[i].a[1];
      varying float zi = A[i].a[2];
        
      varying float dRx = xi;
      dRx -= B[0].a[0];
      varying float dRy = yi;
      dRy -= B[0].a[1];
      varying float dRz = zi;
      dRz -= B[0].a[2];
        
      varying float d1 = EQ;
        
      varying float f0x = 0.0;
      d1 += dRx * dRx;
      varying float f0y = 0.0;
      d1 += dRy * dRy;
      varying float f0z = 0.0;
      d1 += dRz * dRz;
        
      varying float d0 = d1 ;
      varying float p0 = 0.0;
      d0 *= d1 * d1;
      varying float invDist = B[0].SCAL;
      rsqrtNR(d0);
      invDist *= m0 *  d0;
      d0 = invDist * d1;
        
      d1 *=  d0;
        
      dRx *= invDist;
      dRy *= invDist;
      dRz *= invDist;
        
      p0 -=d0;
        
        
      id1 = B[0].id;
      ACPN[id1].POT[0] -= reduce_add(d0);
      f0x -= dRx;
      ACPN[id1].POT[1] += reduce_add(dRx);
      f0y -= dRy;
      varying int id2 = A[i].id;
      ACPN[id1].POT[2] += reduce_add(dRy);
      f0z -= dRz;
      ACPN[id1].POT[3] += reduce_add(dRz);
        
      ACPN[id2].POT[0] += (p0);
        
      ACPN[id2].POT[1] += (f0x);
      ACPN[id2].POT[2] += (f0y);
      ACPN[id2].POT[3] += (f0z);
    }
}
export void cellCellX22(uniform pfalcONstruct A[], uniform pfalcONstruct B[], uniform int ni, uniform int nj, uniform float EQ, uniform pfalcONstructPROP ACPN[])
{
  foreach (i = 0 ... ni)
    {
      uniform int id1 = 0;
      uniform int j = 0;
      uniform int j2 = 1;
      uniform int id12 = 0;
      varying float m0 =  (i < ni) ? A[i].SCAL : 0.0 ;
        
      varying float xi = A[i].a[0];
      varying float yi = A[i].a[1];
      varying float zi = A[i].a[2];
        
      varying float dRx = xi;
      dRx -= B[0].a[0];
      varying float dRy = yi;
      dRy -= B[0].a[1];
      varying float dRz = zi;
      dRz -= B[0].a[2];
        
      varying float d1 = EQ;
        
      varying float f0x = 0.0;
      d1 += dRx * dRx;
      varying float f0y = 0.0;
      d1 += dRy * dRy;
      varying float f0z = 0.0;
      d1 += dRz * dRz;
        
      varying float d0 = d1 ;
      varying float p0 = 0.0;
      d0 *= d1 * d1;
      varying float invDist = B[0].SCAL;
      //


      varying float dRx2 = xi;
      dRx2 -= B[1].a[0];
      varying float dRy2 = yi;
      dRy2 -= B[1].a[1];
      varying float dRz2 = zi;
      dRz2 -= B[1].a[2];
        
      varying float d12 = EQ;
        
      d12 += dRx2 * dRx2;
      d12 += dRy2 * dRy2;
      d12 += dRz2 * dRz2;
        
      varying float d02 = d12 ;
      d02 *= d12 * d12;
      varying float invDist2 = B[1].SCAL;
        
      for(j = 0; j < nj - 2 ; j=j+2, j2=j2+2)
        {
	  rsqrtNR(d0);
	  invDist *= m0  * d0;
	  d0 = invDist * d1;
            
	  id1 = B[j].id;
	  d1 *=  d0;
	  ACPN[id1].POT[0] -= reduce_add(d0);
            
	  dRx *= invDist;
	  ACPN[id1].POT[1] += reduce_add(dRx);
	  f0x -= dRx;
            
	  dRy *= invDist;
	  ACPN[id1].POT[2] += reduce_add(dRy);
	  f0y -= dRy;
            
	  dRz *= invDist;
	  ACPN[id1].POT[3] += reduce_add(dRz);
	  f0z -= dRz;
            
	  d1 = EQ;
            
	  p0 -=d0;
            
	  dRx = xi;
	  dRx -= B[j+2].a[0];
	  d1 += dRx * dRx;
            
	  dRy = yi;
	  dRy -= B[j+2].a[1];
	  d1 += dRy * dRy;
            
	  dRz = zi;
	  dRz -= B[j+2].a[2];
	  d1 += dRz * dRz;
	  d0 = d1;
	  d0 *= d1 * d1;
	  invDist = B[j+2].SCAL;
	  ////
	  rsqrtNR(d02);
	  invDist2 *= m0  * d02;
	  d02 = invDist2 * d12;
            
	  id12 = B[j2].id;
	  d12 *=  d02;
	  ACPN[id12].POT[0] -= reduce_add(d02);
            
	  dRx2 *= invDist2;
	  ACPN[id12].POT[1] += reduce_add(dRx2);
	  f0x -= dRx2;
            
	  dRy2 *= invDist2;
	  ACPN[id12].POT[2] += reduce_add(dRy2);
	  f0y -= dRy2;
            
	  dRz2 *= invDist2;
	  ACPN[id12].POT[3] += reduce_add(dRz2);
	  f0z -= dRz2;
            
	  d12 = EQ;
            
	  p0 -=d02;
            
	  dRx2 = xi;
	  dRx2 -= B[j2+2].a[0];
	  d12 += dRx2 * dRx2;
            
	  dRy2 = yi;
	  dRy2 -= B[j2+2].a[1];
	  d12 += dRy2 * dRy2;
            
	  dRz2 = zi;
	  dRz2 -= B[j2+2].a[2];
	  d12 += dRz2 * dRz2;
	  d02 = d12;
	  d02 *= d12 * d12;
	  invDist2 = B[j2+2].SCAL;
 
	}
      j = nj - 2;
      d0 = d1 * d1 * d1;
      rsqrtNR(d0);
      invDist = m0 * B[j].SCAL * d0;
      d0 = invDist * d1;
        
      d1 *=  d0;
        
      dRx *= invDist;
      dRy *= invDist;
      dRz *= invDist;
        
      p0 -=d0;
        
        
      id1 = B[j].id;
      ACPN[id1].POT[0] -= reduce_add(d0);
      f0x -= dRx;
      ACPN[id1].POT[1] += reduce_add(dRx);
      f0y -= dRy;
      varying int id2 = A[i].id;
      ACPN[id1].POT[2] += reduce_add(dRy);
      f0z -= dRz;
      ACPN[id1].POT[3] += reduce_add(dRz);
      //

      j2 = nj -1;
      d02 = d12 * d12 * d12;
      rsqrtNR(d02);
      invDist2 = m0* B[j2].SCAL * d02;
      d02 = invDist2 * d12;
        
      d12 *=  d02;
        
      dRx2 *= invDist2;
      dRy2 *= invDist2;
      dRz2 *= invDist2;
        
      p0 -=d02;
      id12 = B[j2].id;
      ACPN[id12].POT[0] -= reduce_add(d02);
      f0x -= dRx2;
      ACPN[id12].POT[1] += reduce_add(dRx2);
      f0y -= dRy2;
      ACPN[id12].POT[2] += reduce_add(dRy2);
      f0z -= dRz2;
      ACPN[id12].POT[3] += reduce_add(dRz2);
        
      ACPN[id2].POT[0] += (p0);
      ACPN[id2].POT[1] += (f0x);
      ACPN[id2].POT[2] += (f0y);
      ACPN[id2].POT[3] += (f0z);
    }
}

export void cellCellWoMutualX22(uniform pfalcONstruct A[], uniform pfalcONstruct B[], uniform int ni, uniform int nj, uniform float EQ, uniform pfalcONstructPROP ACPN[])
{
  foreach (i = 0 ... ni)
    {
      uniform int id1 = 0;
      uniform int j = 0;
      uniform int j2 = 1;
      uniform int id12 = 0;
      varying float m0 =  (i < ni) ? A[i].SCAL : 0.0 ;
        
      varying float xi = A[i].a[0];
      varying float yi = A[i].a[1];
      varying float zi = A[i].a[2];
        
      varying float dRx = xi;
      dRx -= B[0].a[0];
      varying float dRy = yi;
      dRy -= B[0].a[1];
      varying float dRz = zi;
      dRz -= B[0].a[2];
        
      varying float d1 = EQ;
        
      varying float f0x = 0.0;
      d1 += dRx * dRx;
      varying float f0y = 0.0;
      d1 += dRy * dRy;
      varying float f0z = 0.0;
      d1 += dRz * dRz;
        
      varying float d0 = d1 ;
      varying float p0 = 0.0;
      d0 *= d1 * d1;
      varying float invDist = (i != j) ? B[j].SCAL : 0.0 ;
      //

      varying float dRx2 = xi;
      dRx2 -= B[j2].a[0];
      varying float dRy2 = yi;
      dRy2 -= B[j2].a[1];
      varying float dRz2 = zi;
      dRz2 -= B[j2].a[2];
        
      varying float d12 = EQ;
        
      d12 += dRx2 * dRx2;
      d12 += dRy2 * dRy2;
      d12 += dRz2 * dRz2;
        
      varying float d02 = d12 ;
      d02 *= d12 * d12;
      varying float invDist2 = (i != j2) ? B[j2].SCAL : 0.0 ;
        
      for(j = 0; j < nj - 2 ; j=j+2, j2=j2+2)
        {
	  rsqrtNR(d0);
	  invDist *= m0  * d0;
	  d0 = invDist * d1;
            
	  id1 = B[j].id;
	  d1 *=  d0;
            
	  dRx *= invDist;
	  f0x -= dRx;
            
	  dRy *= invDist;
	  f0y -= dRy;
            
	  dRz *= invDist;
	  f0z -= dRz;
            
	  d1 = EQ;
            
	  p0 -=d0;
            
	  dRx = xi;
	  dRx -= B[j+2].a[0];
	  d1 += dRx * dRx;
            
	  dRy = yi;
	  dRy -= B[j+2].a[1];
	  d1 += dRy * dRy;
            
	  dRz = zi;
	  dRz -= B[j+2].a[2];
	  d1 += dRz * dRz;
	  d0 = d1;
	  d0 *= d1 * d1;
	  invDist = (i != j+2) ?B[j+2].SCAL : 0.0 ;;
	  ////
	  rsqrtNR(d02);
	  invDist2 *= m0  * d02;
	  d02 = invDist2 * d12;
            
	  id12 = B[j2].id;
	  d12 *=  d02;
            
	  dRx2 *= invDist2;
	  f0x -= dRx2;
            
	  dRy2 *= invDist2;
	  f0y -= dRy2;
            
	  dRz2 *= invDist2;
	  f0z -= dRz2;
            
	  d12 = EQ;
            
	  p0 -=d02;
            
	  dRx2 = xi;
	  dRx2 -= B[j2+2].a[0];
	  d12 += dRx2 * dRx2;
            
	  dRy2 = yi;
	  dRy2 -= B[j2+2].a[1];
	  d12 += dRy2 * dRy2;
            
	  dRz2 = zi;
	  dRz2 -= B[j2+2].a[2];
	  d12 += dRz2 * dRz2;
	  d02 = d12;
	  d02 *= d12 * d12;
	  invDist2 = (i != j2+2) ? B[j2+2].SCAL : 0.0 ;
        }
        
      j = nj - 2;
      d0 = d1 * d1 * d1;
      rsqrtNR(d0);
      invDist = m0 * ((i != j) ? B[j].SCAL : 0.0) * d0;
      d0 = invDist * d1;
        
      d1 *=  d0;
        
      dRx *= invDist;
      dRy *= invDist;
      dRz *= invDist;
        
      p0 -=d0;
        
        
      id1 = B[j].id;
        
      f0x -= dRx;
        
      f0y -= dRy;
      varying int id2 = A[i].id;
        
      f0z -= dRz;
      //
      j2 = nj -1;
      d02 = d12 * d12 * d12;
      rsqrtNR(d02);
      invDist2 = m0* ((i != j2) ? B[j2].SCAL : 0.0) * d02;
      d02 = invDist2 * d12;
        
      d12 *=  d02;
        
      dRx2 *= invDist2;
      dRy2 *= invDist2;
      dRz2 *= invDist2;
        
      p0 -=d02;
      id12 = B[j2].id;
        
      f0x -= dRx2;
        
      f0y -= dRy2;
        
      f0z -= dRz2;
        
        
      ACPN[id2].POT[0] += (p0);
      ACPN[id2].POT[1] += (f0x);
      ACPN[id2].POT[2] += (f0y);
      ACPN[id2].POT[3] += (f0z);
    }

}
export void cellCellWoMutual2(uniform pfalcONstruct A[], uniform pfalcONstruct B[], uniform int ni, uniform int nj, uniform float EQ, uniform pfalcONstructPROP ACPN[])
{
  foreach (i = 0 ... ni)
    {
      uniform int id1 = 0;
      uniform int j = 0;
        
      varying float m0 =  (i < ni) ? A[i].SCAL : 0.0 ;
        
      varying float xi = A[i].a[0];
      varying float yi = A[i].a[1];
      varying float zi = A[i].a[2];
        
      varying float dRx = xi;
      dRx -= B[j].a[0];
      varying float dRy = yi;
      dRy -= B[j].a[1];
      varying float dRz = zi;
      dRz -= B[j].a[2];
        
      varying float d1 = EQ;
        
      varying float f0x = 0.0;
      d1 += dRx * dRx;
      varying float f0y = 0.0;
      d1 += dRy * dRy;
      varying float f0z = 0.0;
      d1 += dRz * dRz;
        
      varying float d0 = d1 ;
      varying float p0 = 0.0;
      d0 *= d1 * d1;
      varying float invDist = ((i != j) ? B[j].SCAL : 0.0);
        
      for(j = 0; j < nj - 1 ; j++)
        {
            
	  rsqrtNR(d0);
	  invDist *= m0  * d0;
	  d0 = invDist * d1;
            
	  id1 = B[j].id;
	  d1 *=  d0;
            
	  dRx *= invDist;
	  f0x -= dRx;
            
	  dRy *= invDist;
            
	  f0y -= dRy;
            
	  dRz *= invDist;
            
	  f0z -= dRz;
            
	  d1 = EQ;
            
	  p0 -=d0;
            
	  dRx = xi;
	  dRx -= B[j+1].a[0];
	  d1 += dRx * dRx;
            
	  dRy = yi;
	  dRy -= B[j+1].a[1];
	  d1 += dRy * dRy;
            
	  dRz = zi;
	  dRz -= B[j+1].a[2];
	  d1 += dRz * dRz;
	  d0 = d1;
	  d0 *= d1 * d1;
	  invDist = ((i != j+1) ? B[j+1].SCAL : 0.0);
            
        }
        
      j = nj -1;
      d0 = d1 * d1 * d1;
      rsqrtNR(d0);
      invDist = m0 * ((i != j) ? B[j].SCAL : 0.0) * d0;
      d0 = invDist * d1;
        
      d1 *=  d0;
        
      dRx *= invDist;
      dRy *= invDist;
      dRz *= invDist;
        
      p0 -=d0;
        
      f0x -= dRx;
      f0y -= dRy;
      f0z -= dRz;
      id1 = B[j].id;
      varying int id2 = A[i].id;
      ACPN[id2].POT[0] += (p0);
        
      ACPN[id2].POT[1] += (f0x);
      ACPN[id2].POT[2] += (f0y);
      ACPN[id2].POT[3] += (f0z);
    }
}

export void cellSelfX22(uniform pfalcONstruct A[], uniform int nd, uniform float EQ, uniform pfalcONstructPROP ACPN[])
{
  uniform int i = 0;
  for(; i < nd- programCount; i+=programCount)
    {
      cellCellX22(A+i,
		  A+i+programCount,
		  programCount,
		  nd - i - programCount ,
		  EQ,ACPN);
      cellCellWoMutualX22(A+i,
                          A+i,
                          programCount,
                          programCount,
                          EQ,ACPN);
    }
  cellCellWoMutual2(A+i,
		    A+i,
		    nd-i,
		    nd-i,
		    EQ,ACPN);

}
