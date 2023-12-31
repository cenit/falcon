//-----------------------------------------------------------------------------+
//                                                                             |
// Monopole.cc                                                                 |
//                                                                             |
// Copyright (C) 2005-2006,2009 Walter Dehnen                                  |
//                                                                             |
// This program is free software; you can redistribute it and/or modify        |
// it under the terms of the GNU General Public License as published by        |
// the Free Software Foundation; either version 2 of the License, or (at       |
// your option) any later version.                                             |
//                                                                             |
// This program is distributed in the hope that it will be useful, but         |
// WITHOUT ANY WARRANTY; without even the implied warranty of                  |
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU           |
// General Public License for more details.                                    |
//                                                                             |
// You should have received a copy of the GNU General Public License           |
// along with this program; if not, write to the Free Software                 |
// Foundation, Inc., 675 Mass Ave, Cambridge, MA 02139, USA.                   |
//                                                                             |
//-----------------------------------------------------------------------------+
//                                                                             |
// Versions                                                                    |
// 0.0    08/02/2005  WD created                                               |
// 0.1    18/02/2005  WD compiled & debugged. seems to work ok.                |
// 0.2    17/05/2005  WD deBUGged (forces were wrong if t0<t<t1)               |
// 0.3    09/08/2006  WD use $NEMOINC/defacc.h                                 |
//-----------------------------------------------------------------------------+
#include <iostream>
#include <fstream>
#include <acceleration.h>
#include <Pi.h>
#include <inline.h>
#include <acc/timer.h>
#define __NO_AUX_DEFACC
#include <defacc.h> // $NEMOINC/defacc.h
#include <stdinc.h>
////////////////////////////////////////////////////////////////////////////////
namespace {
  using namespace WDutils;
  const int AccMax = 10;
  //////////////////////////////////////////////////////////////////////////////
  //                                                                          //
  // class Monopole                                                           //
  //                                                                          //
  // implements an accelerations field generated by the potential             //
  //                                                                          //
  //   P(x,t) = Phi_0(x) + A(t) * [Phi(x) - Phi_0(x)]                         //
  //                                                                          //
  // where Phi(x) is a given potential and Phi_0(x) its monopole. A(t) is     //
  // the adiabatic growth factor from timer.h.                                //
  //                                                                          //
  // NOTES                                                                    //
  //       1. Phi(x) must be a conservative static potential                  //
  //       2. Presently, Phi(x) must be axisymmetric (may be changed in       //
  //          future versions)                                                //
  //                                                                          //
  //////////////////////////////////////////////////////////////////////////////
  class Monopole: private timer {
    //--------------------------------------------------------------------------
    // data                                                                     
    //--------------------------------------------------------------------------
    static const int NT=200;            // # points used in integration d theta 
    int         NR,N1;                  // size of tables below                 
    double      RMIN,RMAX;              // inner and outermost radius           
    double      RQMIN,RQMAX;            // inner and outermost radius squared   
    double     *RQ,*M0,*M1,*M3;         // tables: r, Phi, its 1st, 2nd deriv   
    acc_pter    PHI;                    // conservative external field          
    //--------------------------------------------------------------------------
    // templated acceleration                                                   
    //--------------------------------------------------------------------------
    template <int, typename scalar>
    void acc_T(double       ,                // I: simulation time              
	       int          ,                // I: number bodies =size of arrays
	       const scalar*,                // I: masses:         m[i]         
	       const scalar*,                // I: positions       (x,y,z)[i]   
	       const scalar*,                // I: velocities      (u,v,w)[i]   
	       const int   *,                // I: flags           f[i]         
	       scalar      *,                // O: potentials      p[i]         
	       scalar      *,                // O: accelerations   (ax,ay,az)[i]
	       int          ) const;         // I: add or assign pot & acc?     
    //--------------------------------------------------------------------------
    template <int, typename scalar, bool, bool>
    void set_monopole(scalar       ,
		      int          ,
		      const scalar*,
		      const int   *,
		      scalar      *,
		      scalar      *) const;
    //--------------------------------------------------------------------------
    // destruction                                                              
    //--------------------------------------------------------------------------
  public:
    ~Monopole() {
      if(RQ) delete[] RQ;
      if(M0) delete[] M0;
      if(M1) delete[] M1;
      if(M3) delete[] M3;
    }
    //--------------------------------------------------------------------------
    // construction                                                             
    //--------------------------------------------------------------------------
    Monopole(const double*,
	     int          ,
	     const char  *);
    //--------------------------------------------------------------------------
    // supported call                                                           
    //--------------------------------------------------------------------------
    void acc(int        ndim,                // I: number of dimensions         
	     double     time,                // I: simulation time              
	     int        nbod,                // I: number bodies =size of arrays
	     const void*m,                   // I: masses:         m[i]         
	     const void*x,                   // I: positions       (x,y,z)[i]   
	     const void*v,                   // I: velocities      (u,v,w)[i]   
	     const int *f,                   // I: flags           f[i]         
	     void      *p,                   // O: potentials      p[i]         
	     void      *a,                   // O: accelerations   (ax,ay,az)[i]
	     int        add,                 // I: indicator (see note 6 above) 
	     char       type)                // I: type: 'f' or 'd'             
    {
      switch(type) {
      case 'f':
	switch(ndim) {
	case 2: return acc_T<2>(time,nbod,
				static_cast<const float*>(m),
				static_cast<const float*>(x),
				static_cast<const float*>(v),
				f,
				static_cast<      float*>(p),
				static_cast<      float*>(a),
				add);
	case 3: return acc_T<3>(time,nbod,
				static_cast<const float*>(m),
				static_cast<const float*>(x),
				static_cast<const float*>(v),
				f,
				static_cast<      float*>(p),
				static_cast<      float*>(a),
				add);
	default: error("Monopole: unsupported ndim: %d",ndim);
	}
	break;
      case 'd':
	switch(ndim) {
	case 2: return acc_T<2>(time,nbod,
				static_cast<const double*>(m),
				static_cast<const double*>(x),
				static_cast<const double*>(v),
				f,
				static_cast<      double*>(p),
				static_cast<      double*>(a),
				add);
	case 3: return acc_T<3>(time,nbod,
				static_cast<const double*>(m),
				static_cast<const double*>(x),
				static_cast<const double*>(v),
				f,
				static_cast<      double*>(p),
				static_cast<      double*>(a),
				add);
	default: error("Monopole: unsupported ndim: %d",ndim);
	}
	break;
      default: error("Monopole: unknown type \"%c\"",type);
      }
    }
    //--------------------------------------------------------------------------
  };
  //////////////////////////////////////////////////////////////////////////////
  //                                                                          //
  // auxiliary functions                                                      //
  //                                                                          //
  //////////////////////////////////////////////////////////////////////////////
  void SwallowRestofLine(std::istream& from)
  // swallow rest of line from istream
  {
    char c;
    do from.get(c); while(from.good() && c !='\n');
  }
  //////////////////////////////////////////////////////////////////////////////
  void _read_line(std::istream& from, char*line, int const&n)
  // read line until character `#'; swallow rest of line
  {
    // 1. find first non-space character whereby:
    //    - return empty line if EOF, EOL, '#' are encountered
    do {
      from.get(*line);
      if(from.eof() || (*line)=='\n') {
	*line=0;
	return;
      }
      if((*line)=='#') {
	SwallowRestofLine(from);
	*line=0;
	return;
      }
    } while(isspace(*line));
    // 2. read line until first white-space character
    //    swallow rest til EOL, if necessary 
    register char*l=line+1;
    const    char*L=line+n;
    while(l != L) {
      from.get(*l);
      if(from.eof() || (*l)=='\n') { *l=0; return; }
      if(isspace(*l)) {
	*l=0;
	SwallowRestofLine(from);
	return;
      }
      ++l;
    }
    error("Combined: line longer than expected\n");
  }
  //////////////////////////////////////////////////////////////////////////////
  bool read_line(std::istream& from, char*line, int const&n)
  // read line of size n
  // skip lines whose first non-space character is #
  {
    do {
      _read_line(from,line,n);
    } while(from.good() && *line == 0);
    return *line != 0;
  }
  //////////////////////////////////////////////////////////////////////////////
  template<typename scalar_type>
  int hunt(const scalar_type*xarr, const int n, const scalar_type x,
	   const int j) {
    // hunts the ordered table xarr for jlo such that xarr[jlo]<=x<xarr[jlo+1]  
    // on input j provides a guess for the final value of jlo.                  
    // for an ascendingly ordered array, we return                              
    //  -1 for         x < x[0]                                                 
    //   i for x[i] <= x < x[i+1]  if  0<=i<n                                   
    // n-1 for         x == x[n-1]                                              
    // n   for         x >  x[n-1]                                              
    int  jm,jlo=j,jhi,l=n-1;
    bool ascnd=(xarr[l]>xarr[0]);
    if(!ascnd && xarr[l]==xarr[0] ) return -1;	    // x_0 = x_l                
    if((ascnd && x<xarr[0]) || (!ascnd && x>xarr[0]) ) return -1;
    if((ascnd && x>xarr[l]) || (!ascnd && x<xarr[l]) ) return  n;

    if(jlo<0 || jlo>l) {                            // input guess not useful,  
      jlo = -1;                                     //   go to bisection below  
      jhi = n;
    } else {
      int inc = 1;
      if((x>=xarr[jlo]) == ascnd) {                 // hunt upward              
	if(jlo == l) return (x==xarr[l])? l : n;
	jhi = jlo+1;
	while((x>=xarr[jhi]) == ascnd) {            // not done hunting         
	  jlo =jhi;
	  inc+=inc;                                 // so double the increment  
	  jhi =jlo+inc;
	  if(jhi>l) {                               // off end of table         
	    jhi=n;
	    break;
	  }
	}
      } else {                                      // hunt downward            
	if(jlo == 0) return ascnd? -1 : 0;
	jhi = jlo;
	jlo-= 1;
	while((x<xarr[jlo]) == ascnd) {             // not done hunting         
	  jhi = jlo;
	  inc+= inc;                                // so double the increment  
	  jlo = jhi-inc;
	  if(jlo < 0) {                             // off end of table         
	    jlo = 0;
	    break;
	  }
	}
      }
    }
    while (jhi-jlo != 1) {                          // bisection phase          
      jm=(jhi+jlo) >> 1;
      if((x>=xarr[jm]) == ascnd) jlo=jm;
      else jhi=jm;
    }
    return jlo;
  }
  //----------------------------------------------------------------------------
  template<typename scalar_type>
  inline void find(int& klo, const int n, scalar_type const *x,
		   const scalar_type xi)
  {
    if(klo<0 || klo>=n-1 || x[klo]>xi || x[klo+1]<xi) {
      klo = int( (xi-x[0]) / (x[n-1]-x[0]) * (n-1) );
      klo = hunt(x,n,xi,klo);
      if(klo<0 || klo>=n) 
	error("[%s.%d]: in %s: %s",__FILE__,__LINE__,"x out of range","find()");
    }
  }
  //////////////////////////////////////////////////////////////////////////////
  //                                                                          //
  // Fifth-order splines                                                      //
  //                                                                          //
  // this is a mere collection of (static) methods                            //
  //                                                                          //
  //////////////////////////////////////////////////////////////////////////////
  struct penta_splines {
    //--------------------------------------------------------------------------
    // construct 5th order spline                                               
    //--------------------------------------------------------------------------
    template<class scalar_type, class table_type>
    static void construct(int         const n,	  // I:  size of tables         
			  const scalar_type*x,	  // I:  table: points          
			  const table_type *y,    // I:  table: f(x)            
			  const table_type *y1,   // I:  table: df/dx           
			  table_type *const y3)   // O:  table: d^3f/dx^3       
    {
      const scalar_type zero(0);
      scalar_type p,sig,dx,dx1,dx2;
      table_type  dy=y[1]-y[0], dy1=dy;
      scalar_type *v = new scalar_type[n-1];
      dx   = x[1]-x[0];
      y3[0]= v[0] = 0;
      for(int i=1; i<n-1; i++) {
	dx1  = x[i+1]-x[i];
	dx2  = x[i+1]-x[i-1];
	dy1  = y[i+1]-y[i];
	sig  = dx/dx2;
	p    = sig*v[i-1]-3;
	v[i] = (sig-1)/p;
	y3[i]= times<12>( times< 7>( y1[i]*dx2/(dx*dx1) ) +
			  times< 3>( y1[i-1]/dx+y1[i+1]/dx1 ) -
			  times<10>( dy/(dx*dx) + dy1/(dx1*dx1) ) ) / dx2;
	y3[i]= (y3[i] - sig*y3[i-1] ) / p;
	dx   = dx1;
	dy   = dy1;
      }
      y3[n-1] = zero;
      for(int i=n-2; i>=0; i--)
	y3[i] += v[i]*y3[i+1];
      delete[] v;
    }
    //--------------------------------------------------------------------------
    // evaluate spline at x=xi with xl <= xi <= xh and yl=y(xl) etc...          
    //--------------------------------------------------------------------------
    template<class scalar_type, class table_type>
    static void evaluate(scalar_type const &xi,
			 scalar_type const &x0,
			 scalar_type const &x1,
			 table_type  const &y0,
			 table_type  const &y1,
			 table_type  const &y10,
			 table_type  const &y11,
			 table_type  const &y30,
			 table_type  const &y31,
			 table_type  *yi,
			 table_type  *dyi = 0,
			 table_type  *d2yi= 0)
    {
      const scalar_type ife=1./48.;
      scalar_type h =x1-x0;
      if(h==0) error("penta_splines::evaluate(): bad x input");
      scalar_type
	hi = scalar_type(1)/h,
	hf = h*h,
	A  = hi*(x1-xi), Aq = A*A,
	B  = 1-A,        Bq = B*B,
	C  = h*Aq*B,
	D  =-h*Bq*A;
      table_type
 	t1 = hi*(y1-y0),
	C2 = y10-t1,
	C3 = y11-t1,
	t2 = times<6>(y10+y11-t1-t1)/hf,
	C4 = y30-t2,
	C5 = y31-t2;
      hf *= ife;
      *yi= A*y0+ B*y1+ C*C2+ D*C3+ hf*(C*(Aq+Aq-A-1)*C4+ D*(Bq+Bq-B-1)*C5);
      if(dyi) {
	scalar_type BAA=B-A-A, ABB=A-B-B;
	hf  += hf;
	*dyi = t1 + (A*ABB)*C2 + (B*BAA)*C3
	     + hf*A*B*((1+A-times<5>(Aq))*C4+ (1+B-times<5>(Bq))*C5);
	if(d2yi) {
	  *d2yi = BAA*C2 - ABB*C3;
	  *d2yi+= *d2yi  + hf * ( (twice(Aq)*(times<9>(B)-A)-1) * C4 +
				  (twice(Bq)*(B-times<9>(A))+1) * C5 );
	  *d2yi*= hi;
	}
      }
    }
    //--------------------------------------------------------------------------
    // evaluate penta spline at x=xi                                            
    //--------------------------------------------------------------------------
    template<class scalar_type, class table_type>
    static table_type evaluate(scalar_type const &xi,
			       const scalar_type *x,
			       const table_type  *y,
			       const table_type  *y1,
			       const table_type  *y3,
			       table_type  *dyi = 0,
			       table_type  *d2yi= 0)
    {
      table_type yi;
      evaluate(xi,x[0],x[1],y[0],y[1],y1[0],y1[1],y3[0],y3[1],&yi,dyi,d2yi);
      return yi;
    }
    //--------------------------------------------------------------------------
    // spline evaluation                                                        
    //--------------------------------------------------------------------------
    template<class scalar_type, class table_type>
    static table_type eval (int               n,
			    scalar_type const&xi,
			    const scalar_type*x,
			    const table_type *y,
			    const table_type *y1,
			    const table_type *y3,
			    table_type       *dy = 0,
			    table_type       *d2y= 0)
    {
      static int lo=0;
      find(lo,n,x,xi);
      if(lo==n-1) --lo;
      return evaluate(xi,x+lo,y+lo,y1+lo,y3+lo,dy,d2y);
    }
  };
  //////////////////////////////////////////////////////////////////////////////
  //                                                                          //
  // Gauss-Legendre integration: points and weights                           //
  //                                                                          //
  //////////////////////////////////////////////////////////////////////////////
  void GaussLegendre(double *x, double *w, const unsigned n)
  {
    const double eps=std::numeric_limits<double>::epsilon();
    const int m=(n+1)/2;
    for(int i=0; i!=m; ++i) {
      double z1,pp,z=std::cos(Pi*(i+0.75)/(n+0.5));
      do {
	double p1 = 1.0;
	double p2 = 0.0;
	for(unsigned j=0; j!=n; ++j) {
	  double p3 = p2;
	  p2 = p1;
	  p1 = ( (2*j+1)*z*p2 - j*p3 ) / double(j+1);
	}
	pp = n * (z*p1-p2) / (z*z-1.0);
	z1 = z;
	z  = z1 - p1 / pp;
      } while (abs(z-z1)>eps);
      x[i]     =-z;
      x[n-1-i] = z;
      w[i]     = 2. / ((1.0-z*z)*pp*pp);
      w[n-1-i] = w[i];
    }
  }
  //////////////////////////////////////////////////////////////////////////////
  //                                                                          //
  // auxiliary templates                                                      //
  //                                                                          //
  //////////////////////////////////////////////////////////////////////////////
  template <typename scalar> struct _type;
  template <> struct _type<float > { static const char t='f'; };
  template <> struct _type<double> { static const char t='d'; };
  //////////////////////////////////////////////////////////////////////////////
  template <int, bool> struct set;
  template <int NDIM> struct set<NDIM,0> {
    template<typename X, typename Y>
    static void s(X&a, Y const&b) { a = b; }
    template<typename X, typename Y>
    static void v(X*a, const Y*b, Y const&x) { v_asstimes<NDIM>(a,b,x); }
  };
  template <int NDIM> struct set<NDIM,1> {
    template<typename X, typename Y>
    static void s(X&a, Y const&b) { a += b; }
    template<typename X, typename Y>
    static void v(X*a, const Y*b, Y const&x) { v_addtimes<NDIM>(a,b,x); }
  };
  //////////////////////////////////////////////////////////////////////////////
  //                                                                          //
  // class Monopole                                                           //
  //                                                                          //
  //////////////////////////////////////////////////////////////////////////////
  Monopole::Monopole(const double*pars,
		     int          npar,
		     const char  *file) :
    // initialize base and data
    timer ( timer::adiabatic,
	    npar>0? pars[0] : 0.,
	    npar>1? pars[1] : 1. ),
    NR    ( npar>4? int(pars[4]) : 1001 ),
    N1    ( NR - 1 ),
    RMIN  ( npar>2? pars[2] : 0.001 ),
    RMAX  ( npar>3? pars[3] : 1000. ),
    RQMIN ( square( RMIN ) ),
    RQMAX ( square( RMAX ) ),
    RQ    ( NR>1? new double[NR] : 0 ),
    M0    ( NR>1? new double[NR] : 0 ),
    M1    ( NR>1? new double[NR] : 0 ),
    M3    ( NR>1? new double[NR] : 0 )
  {
    //
    // 0. check consistency of input
    //
    if(npar<5 && nemo_debug(2))
      std::cerr<<"### nemo Debug info: Monopole:\n"
	"  provides the acceleration field due to the potential\n\n"
	"  Phi_0(x) + A(t) * [Phi(x) - Phi_0(x)],\n\n"
	"  where Phi(x) is a given conservative potential, Phi_0(x) its\n"
	"  monopole and A(t) an adiabatic growth factor with A=0 at t<t0\n"
	"  and A=1 at t>t0+tau.\n"
	"  A data file is required and must be of the form\n"
	"    accname=NAME\n"
	"   [accpars=PARS]\n"
	"   [accfile=FILE]\n"
	"  characters after (and including) '#' are ignored.\n"
	"  Parameters: 5 with the meanings:\n"
	"  par[0] = t0:  start time for growth                   [0]\n"
	"  par[1] = tau: time scale for growth                   [1]\n"
	"  par[2] = innermost radius in table for monopole       [0.001]\n"
	"  par[3] = outermost radius in table for monopole       [1000]\n"
	"  par[4] = number of points in table for monopole       [1001]\n\n";
    if(npar>5)
      warning("Monopole: skipped parameters beyond 6\n");
    nemo_dprintf(4,"Monopole: timer set to: %s with t0=%f, tau=%f\n",
		 timer::describe(), timer::T0(), timer::TAU());
    if(NR < 2)
      error("Monopole: NR=%d < 2\n",NR);
    if(NR < 10)
      warning("Monopole: NR=%d < 10\n",NR);
    if(file==0|| file[0]==0)
      error("Monopole: no accfile given\n");
    //
    // 1. scan data file for accname, accpars, accfile, and initialize Phi
    //
    nemo_dprintf(4,"Monopole: scanning file \"%s\"\n",file);
    const int size=256;
    std::ifstream inpt(file);
    if(! inpt.good())
      error("Monopole: couldn't open file \"%s\" for input\n",file);
    char Line[size], AccName[size], AccPars[size], AccFile[size];
    char*accname=0, *accpars=0, *accfile=0, unknown[9];
    if(! read_line(inpt,Line,size))
      error("Monopole: couldn't read data from file \"%s\"\n",file);
    do {
      if       (0==strncmp(Line,"accname=",8)) {
	if(accname)
	  error("Monopole: >1 accname= entry in file \"%s\"\n",file);
	else {
	  strcpy(AccName,Line+8);
	  accname=AccName;
	}
      } else if(0==strncmp(Line,"accpars=",8)) {
	if(accpars)
	  warning("Monopole: extra accpars= in file \"%s\" ignored\n",file);
	else {
	  strcpy(AccPars,Line+8);
	  accpars=AccPars;
	}
      } else if(0==strncmp(Line,"accfile=",8)) {
	if(accfile)
	  warning("Monopole: extra accfile= in file \"%s\" ignored\n",file);
	else {
	  strcpy(AccFile,Line+8);
	  accfile=AccFile;
	}
      } else {
	strncpy(unknown,Line,8); unknown[8]=0;
	warning("Monopole: entry \"%s\" in file \"%s\" ignored\n",
		unknown,file);
      }
    }
    while(read_line(inpt,Line,size));
    if(accname == 0)
      error("Monopole: no accname= entry found in file \"%s\"\n",file);
    nemo_dprintf(4,"Monopole: successfully scanned accfile:\n");
    nemo_dprintf(4,"	       accname=%s\n",accname);
    if(accpars) nemo_dprintf(4,"	       accpars=%s\n",accpars);
    if(accfile) nemo_dprintf(4,"	       accfile=%s\n",accfile);
    nemo_dprintf(4,"Monopole: now initializing \"%s\"\n",accname);
    bool m(0),v(0);
    PHI = get_acceleration(accname,accpars,accfile,&m,&v);
    if(PHI==0)
      error("Monopole: cannot obtain conservative potential\n");
    if(m || v)
      error("Monopole: Phi(x) not conservative\n");
    //
    // 2. initialize tables and compute monopole
    //
    // 2.1 set table with radii^2; also make a temporary table with radii
    //
    double *R = new double[  NR];
    double dL = log(RMAX/RMIN) / double(N1);
    for(int i=0; i!=NR; ++i) {
      R [i] = RMIN * exp(i*dL);
      RQ[i] = square(R[i]);
    }
    R [0]  = RMIN;
    RQ[0]  = RQMIN;
    R [N1] = RMAX;
    RQ[N1] = RQMAX;
    //
    // 2.2 obtain Gauss-Legendre weights for integration over cos(theta)
    //
    double *S = new double[NT];
    double *C = new double[NT];
    double *W = new double[NT];
    GaussLegendre(C,W,NT);
    for(int j=0; j!=NT; ++j) {
      S[j] = sqrt(1.-C[j]*C[j]);
      W[j]*= 0.5;
    }
    //
    // 2.3 set positions to (x=r*sin(theta), y=0, z=r*cos(theta))
    //
    const int N=NR*NT;
    double *X = new double[3*N];
    double *x = X;
    for(int i=0; i!=NR; ++i)
      for(int j=0; j!=NT; ++j, x+=3) {
	x[0] = R[i] * S[j];
	x[1] = 0;
	x[2] = R[i] * C[j];
      }
    //
    // 2.4 call Phi to get potential and forces
    //
    double *P = new double[  N];
    double *A = new double[3*N];
    PHI(3,0.,N,0,X,0,0,P,A,0,'d');
    //
    // 2.5 integrate over cos(theta) and assign M0, M1 = dM0/d(R^2)
    //
    double *p = P;
    double *a = A;
    for(int i=0; i!=NR; ++i) {
      double m0=0., m1=0.;
      for(int j=0; j!=NT; ++j, p++,a+=3) {
	m0 += W[j] * p[0];
	m1 += W[j] *(S[j] * a[0] + C[j] * a[2]);
      }
      M0[i] = m0;                     // M0 = Monopole potential
      M1[i] =-0.5 * m1 / R[i];        // d M0 / d(R^2)
    }
    //
    // 2.6 create fifth-order spline and  delete temporary arrays
    //
    penta_splines::construct(NR,RQ,M0,M1,M3);
    delete[] W;
    delete[] C;
    delete[] S;
    delete[] R;
    delete[] X;
    delete[] P;
    delete[] A;
  }
  //////////////////////////////////////////////////////////////////////////////
  template <int NDIM, typename scalar, bool ADD_P, bool ADD_A>
  void Monopole::set_monopole(scalar       ampl,
			      int          nbod,
			      const scalar*x,
			      const int   *f,
			      scalar      *p,
			      scalar      *a) const
  {
    const scalar pmono = 1-ampl, fmono = -2*pmono;
    nemo_dprintf(4,"Monopole: setting P = %f * Phi_0\n",pmono);
    for(int n=0; n!=nbod; ++n,x+=NDIM,p++,a+=NDIM)
      if(f==0 || f[n] & 1) {
	double xq = v_norm<NDIM>(x), f0, p0;
	if(xq > RQMAX) {
	  double r = sqrt(RQMAX/xq);
	  p0 = M0[N1] * r;
	  f0 =-0.5 * p0 / xq;
	} else if(xq < RQMIN) {
	  p0 = M0[0] + (xq-RQMIN) * M1[0];
	  f0 = M1[0];
	} else
	  p0 = penta_splines::eval(NR,xq,RQ,M0,M1,M3,&f0);
	set<NDIM,ADD_P>::s(p[0], pmono*p0);
	set<NDIM,ADD_A>::v(a,x,fmono*scalar(f0));
      }
  }
  //////////////////////////////////////////////////////////////////////////////
  template <int NDIM, typename scalar>
  void Monopole::acc_T(double       time,
		       int          nbod,
		       const scalar*,
		       const scalar*x,
		       const scalar*,
		       const int   *f,
		       scalar      *p,
		       scalar      *a,
		       int          add) const
  {
    // obtain amplitude of growth factor
    const scalar ampl = timer::operator()(time);
    nemo_dprintf(4,"Monopole: amplitude=%f\n",ampl);
    //
    // 1. if A(t) == 1 external potential only
    //
    if(ampl == 1.) {
      nemo_dprintf(4,"Monopole: setting P = Phi\n");
      PHI(NDIM,time,nbod,0,x,0,f,p,a,add,_type<scalar>::t);
      return;                                                     // DONE !
    }
    //
    // 2. set/add [1-A(t)] times monopole
    //
    if(add & 1)
      if(add & 2) set_monopole<NDIM,scalar,1,1>(ampl,nbod,x,f,p,a);
      else        set_monopole<NDIM,scalar,1,0>(ampl,nbod,x,f,p,a);
    else
      if(add & 2) set_monopole<NDIM,scalar,0,1>(ampl,nbod,x,f,p,a);
      else        set_monopole<NDIM,scalar,0,0>(ampl,nbod,x,f,p,a);
    if(ampl == 0.) return;                                        // DONE !
    //
    // 3. add A(t) times external potential
    //
    nemo_dprintf(4,"Monopole: adding P += %f * Phi\n",ampl);
    scalar *pots = new scalar[nbod];
    scalar *accs = new scalar[NDIM*nbod];
    PHI(NDIM,time,nbod,0,x,0,f,pots,accs,0,_type<scalar>::t);
    scalar       *ps = pots;
    scalar       *as = accs;
    for(int n=0; n!=nbod; ++n,p++,ps++,a+=NDIM,as+=NDIM) 
      if(f==0 || f[n] & 1) {
	p[0] += ampl * ps[0];
	v_addtimes<NDIM>(a,as,ampl);
      }
    delete[] pots;
    delete[] accs;
  }
  //////////////////////////////////////////////////////////////////////////////
  //                                                                          //
  // auxiliary for iniacceleration()                                          //
  //                                                                          //
  //////////////////////////////////////////////////////////////////////////////
  Monopole *MyAcc[AccMax] = {0};
  int       AccN          = 0;
#undef  _DEF_ACC_NO
#define _DEF_ACC_NO(NUM)				\
  void acceleration##NUM(int        d,			\
			 double     t,			\
			 int        n,			\
			 const void*m,			\
			 const void*x,			\
			 const void*v,			\
			 const int *f,			\
			 void      *p,			\
			 void      *a,			\
			 int        i,			\
			 char       y)			\
    { (MyAcc[NUM])->acc(d,t,n,m,x,v,f,p,a,i,y); }
  _DEF_ACC_NO(0)
  _DEF_ACC_NO(1)
  _DEF_ACC_NO(2)
  _DEF_ACC_NO(3)
  _DEF_ACC_NO(4)
  _DEF_ACC_NO(5)
  _DEF_ACC_NO(6)
  _DEF_ACC_NO(7)
  _DEF_ACC_NO(8)
  _DEF_ACC_NO(9)
  acc_pter Accs[AccMax] = {&acceleration0,
			   &acceleration1,
			   &acceleration2,
			   &acceleration3,
			   &acceleration4,
			   &acceleration5,
			   &acceleration6,
			   &acceleration7,
			   &acceleration8,
			   &acceleration9};
} // namespace {
////////////////////////////////////////////////////////////////////////////////
void iniacceleration(const double*pars,      // I:  array with parameters       
		     int          npar,      // I:  number of parameters        
		     const char  *file,      // I:  data file name              
		     acc_pter    *accel,     // O:  pter to acceleration()      
		     bool        *needM,     // O:  acceleration() needs masses?
		     bool        *needV)     // O:  acceleration() needs vel's? 
{
  if(AccN == AccMax) {
    warning("iniacceleration(): request to initialize "
	    "more than %d accelerations of type \"Monopole\"", AccMax);
    *accel = 0;
    return;
  }
  MyAcc[AccN] = new Monopole(pars,npar,file);
  if(needM) *needM = false;
  if(needV) *needV = false;
  *accel = Accs[AccN++];
}
////////////////////////////////////////////////////////////////////////////////
