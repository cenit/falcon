//-----------------------------------------------------------------------------+
//                                                                             |
// Combined.cc                                                                 |
//                                                                             |
// Copyright (C) 2004-2006 Walter Dehnen                                       |
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
// 0.0    17/08/2004  WD created based on combined.cc                          |
// 1.0    24/08/2004  WD allow for up to 10 instantinations                    |
// 1.1    09/08/2006  WD use $NEMOINC/defacc.h                                 |
//-----------------------------------------------------------------------------+
#include <iostream>
#include <fstream>
#include <acceleration.h>
#include <acc/timer.h>
#define  __NO_AUX_DEFACC
#include <stdinc.h>
#include <defacc.h>      // $NMOINC/defacc.h
////////////////////////////////////////////////////////////////////////////////
namespace {
  //----------------------------------------------------------------------------
  const int AccMax = 10;
  //----------------------------------------------------------------------------
  template <typename scalar> struct _type;
  template <> struct _type<float > { static const char t='f'; };
  template <> struct _type<double> { static const char t='d'; };
  //////////////////////////////////////////////////////////////////////////////
  //                                                                          //
  // class Combined                                                           //
  //                                                                          //
  // implements an accelerations field generated by the potential             //
  //                                                                          //
  // A(t) * Sum_i Phi_i(x,t)                                                  //
  //                                                                          //
  // where A(t) is one of the growth factors in timer.h and Phi_i are up to   //
  // NMAX = 10 external potentials/acceleration fields.                       //
  //                                                                          //
  //////////////////////////////////////////////////////////////////////////////
  class Combined : private timer {
    static void SwallowRestofLine(std::istream& from)
      // swallow rest of line from istream
    {
      char c;
      do from.get(c); while(from.good() && c !='\n');
    }
    //--------------------------------------------------------------------------
    static void _read_line(std::istream& from, char*line, int const&n)
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
    //--------------------------------------------------------------------------
    static void read_line(std::istream& from, char*line, int const&n)
      // read line of size n
      // skip lines whose first non-space character is #
    {
      do {
	_read_line(from,line,n);
      } while(from.good() && *line == 0);
    }
    //--------------------------------------------------------------------------
    static const int NMAX=10;                 // max number of potentials       
    int              N;                       // actual number of potentials    
    acc_pter         AC[NMAX];                // pointers to accelerations      
    bool             NEEDM,NEEDV;             // need masses, velocities?       
    //--------------------------------------------------------------------------
    template <int NDIM, typename scalar> inline
    void acc_T(double       time,            // I: simulation time              
	       int          nbod,            // I: number bodies =size of arrays
	       const scalar*m,               // I: masses:         m[i]         
	       const scalar*x,               // I: positions       (x,y,z)[i]   
	       const scalar*v,               // I: velocities      (u,v,w)[i]   
	       const int   *f,               // I: flags           f[i]         
	       scalar      *p,               // O: potentials      p[i]         
	       scalar      *a,               // O: accelerations   (ax,ay,az)[i]
	       int          add)             // I: add or assign pot & acc?     
    {
      // obtain amplitude of growth factor
      scalar ampl = timer::operator()(time);

      // if amplitude == 0, set pot / acc to zero if assigning required.
      if(ampl == 0.) {
	if(! (add&1))
	  for(int n=0; n!=nbod; ++n)
	    if(f==0 || f[n] & 1) 
	      p[n] = scalar(0);
	if(! (add&2))
	  for(int n=0,nn=0; n!=nbod; ++n,nn+=NDIM)
	    if(f==0 || f[n] & 1) 
	      v_set<NDIM>(a+nn,scalar(0));
	return;
      }
    
      // if amplitude != 0, must compute accelerations:
      // if amplitude != 1, create arrays to write pot & acc into
      scalar*_pot = ampl!=1 ? new scalar[nbod]      : 0;
      scalar*_acc = ampl!=1 ? new scalar[nbod*NDIM] : 0;

      // define references to the arrays actually passed to accelerations
      scalar*&pots = ampl!=1 ? _pot : p;
      scalar*&accs = ampl!=1 ? _acc : a;

      // add/assign gravity from all the acceleration fields
      for(int i=0; i<N; ++i)
	(*(AC[i]))(NDIM,time,nbod,
		   static_cast<const void*>(m),
		   static_cast<const void*>(x),
		   static_cast<const void*>(v),
		   f,
		   static_cast<      void*>(pots),
		   static_cast<      void*>(accs),
		   i    != 0? 3 :           // further? add pot & acc           
		   ampl != 1? 0 : add,      // first? need to mul? ass:input    
		   _type<scalar>::t);

      // if amplitude != 1, multiply gravity by amplitude
      if(ampl!=1) {

	// add or assign potential times amplitude
	if(add & 1) {
	  for(int n=0; n!=nbod; ++n)
	    if(f==0 || f[n] & 1) 
	      p[n] += ampl * pots[n];
	} else {
	  for(int n=0; n!=nbod; ++n)
	    if(f==0 || f[n] & 1) 
	      p[n]  = ampl * pots[n];
	}
	delete[] _pot;

	// add or assign acceleration times amplitude
	if(add & 2) {
	  for(int n=0,nn=0; n!=nbod; ++n,nn+=NDIM)
	    if(f==0 || f[n] & 1)
	      v_addtimes<NDIM>(a+nn, accs+nn, ampl);
	} else {
	  for(int n=0,nn=0; n!=nbod; ++n,nn+=NDIM)
	    if(f==0 || f[n] & 1)
	      v_asstimes<NDIM>(a+nn, accs+nn, ampl);
	}
	delete[] _acc;
      }
    }
    //--------------------------------------------------------------------------
  public:
    static const char* name() { return "Combined"; }
    bool const&NeedMass() const { return NEEDM; }
    bool const&NeedVels() const { return NEEDV; }
    //--------------------------------------------------------------------------
    Combined(const double*pars,
	     int          npar,
	     const char  *file) : NEEDM(0), NEEDV(0)
    {
      if(npar < 4 || file==0)
	warning("Combined: recognizes 3 parameters and requires a data file.\n"
		"parameters:\n"
		"  par[0] = controlling growth factor, see below         [9]\n"
		"  par[1] = t0: start time for growth                    [0]\n"
		"  par[2] = tau: time scale for growth                   [1]\n"
		"  par[3] = t1: end time for growth (only for exp)      [10]\n"
		"  with par[0]=0: %s\n"
		"       par[0]=1: %s\n"
		"       par[0]=2: %s\n"
		"       par[0]=3: %s\n"
		"       par[0]=4: %s\n"
		"       par[0]=9: %s\n"
		"the data file must contain up to %d entries of the form\n"
		"  accname=ACCNAME\n [accpars=ACCPARS]\n [accfile=ACCFILE]\n"
		"where [] indicates optional entries. Data between a '#' and\n"
		"end-of-line are ignored (allowing comments)\n\n",
		timer::describe(timer::adiabatic),
		timer::describe(timer::saturate),
		timer::describe(timer::quasi_linear),
		timer::describe(timer::linear),
		timer::describe(timer::exponential),
		timer::describe(timer::constant),
		NMAX);
      if(file == 0) error("Combined: not data file given");
      // initialize timer
      timer::index
	timin = (timer::index)(npar>0? int(pars[0]) : 9);
      double
	_t0    = npar>1? pars[1] : 0.,
	_tau   = npar>2? pars[2] : 1.,
	_t1    = npar>3? pars[3] : 10.;
      timer::init(timin,_t0,_tau,_t1);
      if(npar>4) warning("Combined: skipped parameters beyond 4");
      nemo_dprintf (1,
		    "initializing Combined\n"
		    " parameters : timer::index  = %f -> %s\n"
		    "              t0            = %f\n"
		    "              tau           = %f\n"
		    "              t1            = %f\n",
		    timin,timer::describe(timin),_t0,_tau,_t1);
      // now scan datafile and initialize accs
      const int size=200;
      std::ifstream inpt(file);
      if(!inpt.good())
	error("Combined: couldn't open file \"%s\" for input\n",file);
      char Line[size], AccName[size], AccPars[size], AccFile[size];
      char*accname=0, *accpars=0, *accfile=0, unknown[9];
      read_line(inpt,Line,size);
      if(*Line == 0)
	error("Combined: couldn't read data from file \"%s\"\n",file);
      if(strncmp(Line,"accname=",8))
	error("Combined: first entry in file \"%s\" isn't \"accname=...\"\n",
	      file);
      N = 0;
      nemo_dprintf (1," sub-potentials:\n");
      do {
	if(accname==0) {
	  strcpy(AccName,Line+8);
	  accname=AccName;
	  accpars=0;
	  accfile=0;
	}
	read_line(inpt,Line,size);
	if        (*Line == 0 || 0==strncmp(Line,"accname=",8)) {
	  if(N == NMAX)
	    error("Combined: file \"%s\" contains more accname= "
		  "than anticipated\n",file);
	  //       // TEST
	  //       std::cerr<<"loading acceleration "<<N<<": \n"
	  // 	       <<"  accname="<<accname<<'\n';
	  //       if(accpars) std::cerr<<"  accpars="<<accpars<<'\n';
	  //       if(accfile) std::cerr<<"  accfile="<<accfile<<'\n';
	  //       // tensor_set
	  nemo_dprintf (1,"   accname=%s",accname);
	  if(accpars) nemo_dprintf (1," accpars=%s",accpars);
	  if(accfile) nemo_dprintf (1," accfile=%s",accfile);
	  nemo_dprintf (1,"\n");
	  bool nm,nv;
	  AC[N] = get_acceleration(accname,accpars,accfile,&nm,&nv);
	  if(nm) NEEDM = 1;
	  if(nv) NEEDV = 1;
	  N++;
	  accname=0;
	} else if(0==strncmp(Line,"accpars=",8)) {
	  if(accpars)
	    warning("Combined: additional \"accpars=\""
		    " in file \"%s\" ignored\n",file);
	  else {
	    strcpy(AccPars,Line+8);
	    accpars = AccPars;
	  }
	} else if(0==strncmp(Line,"accfile=",8)) {
	  if(accfile)
	    warning("Combined: additional \"accfile=\""
		    " in file \"%s\" ignored\n",file);
	  else {
	    strcpy(AccFile,Line+8);
	    accfile = AccFile;
	    if(0==strcmp(file,accfile))
	      error("Combined: recursion accfile=datafile not allowed\n");
	  }
	} else {
	  strncpy(unknown,Line,8); unknown[8]=0;
	  warning("Combined: entry \"%s\" in file \"%s\" ignored\n",
		  unknown,file);
	}
      } while(*Line != 0);
    } // Combined::Combined()
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
	default: error("Combined: unsupported ndim: %d",ndim);
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
	default: error("Combined: unsupported ndim: %d",ndim);
	}
	break;
      default: error("Combined: unknown type \"%c\"",type);
      }
    } // Combined::acc()
  } *MyAcc[AccMax] = {0};
  int AccN = 0;
  
#undef  _DEF_ACC_NO
#define _DEF_ACC_NO(NUM)					\
void acceleration##NUM(int        d,				\
		       double     t,				\
		       int        n,				\
		       const void*m,				\
		       const void*x,				\
		       const void*v,				\
		       const int *f,				\
		       void      *p,				\
		       void      *a,				\
		       int        i,				\
		       char       y)				\
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
	    "more than %d accelerations of type \"Combined\"", AccMax);
    *accel = 0;
    return;
  }
  MyAcc[AccN] = new Combined(pars,npar,file);
  if(needM) *needM = (MyAcc[AccN])->NeedMass();
  if(needV) *needV = (MyAcc[AccN])->NeedVels();
  *accel = Accs[AccN++];
}
////////////////////////////////////////////////////////////////////////////////
