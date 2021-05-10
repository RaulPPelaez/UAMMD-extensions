/* 
   Solves the Brownian rotation and translation of spherical particles.
   The translational part follows the Euler Maruyama scheme of BrownyanDynamics.cuh
   
   The rotational part is solved using quaternions and their properties to encode
   the orientations and the rotations of the particles. [1]
   
   This file needs ParticleData to provide support for torques and directions
   The quaternion that encodes the orientation the particle i is a real4 contained
   in dir[i]
 
   Translational differential equation:
   X[t+dt] = X[t] + dt·Mt·F[t] + sqrt(2*Tdt)·dW·Bt
   Being:
     X - Positions
     Mt - Translational Self Diffusion  coefficient -> 1/(6·pi·vis·radius)
     dW- Noise vector
     Bt - sqrt(Mt)
   
   Rotational differential equation:
   dphi = dt(Mr·Tor[t])+sqrt(2·T·dt)·dW·Br
   Q[t+dt] = dQ*Q[t+dt]
   Being:
      dphi - Rotation vector
      dQ - Quaternion encoding the rotation vector
      *  - Product between quaternions
      Q  - Quaternion encoding the orientation of the particles
      Mr - Rotational Self Diffusion coefficient -> 1/(8·pi·vis·radius^3)
      Br - sqrt(Mr)
   
   Given two quaternions q1(n1,v1) and q2(n2,v2) the product q3(n3,v3) = q1*q2 is defined as:
   q3 = q1*q2 = (n1·n2 - v1·v2, n1·v2 + n2·v1 + v1 x v2)

   Both differential equations are solved using the Euler-Maruyama scheme

   References:
     [1] https://aip.scitation.org/doi/10.1063/1.4932062

   Contributors:
     - Raul P. Pelaez     -> Translational part
     - P. Palacios Alonso -> Rotational part
*/

#include "Integrator/Integrator.cuh"
#include <curand.h>
#include <thrust/device_vector.h>
namespace uammd{
  namespace extensions{
    namespace BDR{
      struct Parameters{
	std::vector<real3> K = std::vector<real3>(3,real3());
	real temperature = 0;
	real viscosity = 1.0;
	real hydrodynamicRadius = -1.0;
	real dt = 0.0;
      };
    
      class BrownianRotation: public Integrator{
      public:
	using Parameters = BDR::Parameters;
      
	BrownianRotation(shared_ptr<ParticleData> pd,
			 shared_ptr<ParticleGroup> pg,
			 shared_ptr<System> sys,
			 Parameters par);
	
	BrownianRotation(shared_ptr<ParticleData> pd,
			 shared_ptr<System> sys,
			 Parameters par):
	  BrownianRotation(pd, std::make_shared<ParticleGroup>(pd, sys),sys, par){}
      
	~BrownianRotation();
      
	virtual void forwardTime() override;

	virtual real sumEnergy() override{
	  //Sum 1.5*kT to each particle
	  auto energy = pd->getEnergy(access::gpu, access::readwrite);
	  auto energy_gr = pg->getPropertyIterator(energy);
	  auto energy_per_particle = thrust::make_constant_iterator<real>(1.5*temperature);
	  thrust::transform(thrust::cuda::par,
			    energy_gr, energy_gr + pg->getNumberParticles(),
			    energy_per_particle,
			    energy_gr,
			    thrust::plus<real>());
	  return 0;
	}


      protected:
	real3 Kx, Ky, Kz; //shear matrix
	real dt;
	real rotSelfMobility;
	real translSelfMobility;
	real hydrodynamicRadius;
	cudaStream_t stream;
	int steps;   
	uint seed;
	real temperature;
  
	void updateInteractors();
	void resetForces();
	void resetTorques();
	void computeCurrentForces();   
	void updatePositions();
	real* getParticleRadiusIfAvailable();
      };
    }
  }
}
#include"BrownianRotation.cu"
