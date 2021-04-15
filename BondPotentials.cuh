/* Raul P. Pelaez 2021. Bond potentials extensions. 
Place in this file new bond potentials that you have developed.

You are encouraged to add your name and contribution here below.

Contributors:

  -Raul P. Pelaez: HarmonicBond

 */
#include<uammd.cuh>

namespace uammd{
  namespace extensions{

    //You are free (and encouraged) to nest your extension in another namespace if you see it fit:
    //namespace bonds{
    //HarmonicBond is already present in uammd, but we will add it here as an example extension:
    struct HarmonicBond{
      //Place in this struct whatever static information is needed for a given bond
      //In this case spring constant and equilibrium distance
      //the function readBond below takes care of reading each BondInfo from the file
      struct BondInfo{
	real k, r0;
      };
      //This function will be called for every bond read in the bond file
      //In the case of a Fixed Point bond, j will be -1
      //i,j: id of particles in bond
      //r12: ri-rj
      //bi: bond information.
      inline __device__ real3 force(int i, int j, real3 r12, BondInfo bi){
	real r2 = dot(r12, r12);
	if(r2==real(0.0)) return make_real3(0.0);
	real invr = rsqrt(r2);
	real f = -bi.k*(real(1.0)-bi.r0*invr); //F = -k·(r-r0)·rvec/r
	return f*r12;
      }
      inline __device__ real energy(int i, int j, real3 r12, BondInfo bi){
	real r2 = dot(r12, r12);
	if(r2==real(0.0)) return real(0.0);
	real r = sqrt(r2);
	const real dr = r-bi.r0;
	return real(0.5)*bi.k*dr*dr;
      }

      //This function will be called for each bond in the bond file
      //It must use the stream that is handed to it to construct a BondInfo.  
      static __host__ BondInfo readBond(std::istream &in){
	/*BondedForces will read i j, readBond has to read the rest of the line*/
	BondInfo bi;
	in>>bi.k>>bi.r0;
	return bi;
      }

    };

  }
}
