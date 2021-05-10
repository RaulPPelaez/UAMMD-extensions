/* P. Palacios Alonso 2021
Some quaternion algebra and useful functions
Notation:
   n(real)  - Scalar part 
   v(real3) - vectorial part
   q(real4) - quaternion
 */

namespace uammd{
  namespace extensions{
    namespace quaternion{
      
      __host__ __device__ real4 make_quaternion(real n, real3 v){
	return make_real4(n,v.x,v.y,v.z);
      }

      __host__ __device__ real3 getVector(real4 q){
	// Returns the vectorial part of the quaternion
	return make_real3(q.y, q.z, q.w);
      }

      __host__ __device__ real4 rotVec2Quaternion(real3 vrot, real ang){
	/* Returns the quaternion that encondes a rotation of ang radians 
	   around the axis vrot 	 
	   q = (cos(ang/2),vrot·sin(ang/2))
	*/
	vrot*=rsqrt(dot(vrot,vrot)); // The rotation axis must be a unitary vector
	real cang2, sang2;
	real* cang2_ptr = &cang2;
	real* sang2_ptr = &sang2;
	sincos(ang*0.5,sang2_ptr,cang2_ptr);
	real4 q = make_quaternion(cang2,sang2*vrot);
	return q;
      }

      __host__ __device__ real4 rotVec2Quaternion(real3 vrot){
	// If no angle is given the rotation angle is the modulus of vrot
	real ang = sqrt(dot(vrot,vrot));
	return rotVec2Quaternion(vrot,ang);
      }
      
      __host__ __device__ real4 prod (real4 q1, real4 q2){
	/*
	  Product of two quaternions:
	  q3 = q1*q2 = (n1*n2 - v1*v2, n1*v2 + n2*v1 + v1 x v2)
	*/      
	real n1 = q1.x;
	real n2 = q2.x;
	real3 v1 = getVector(q1);
	real3 v2 = getVector(q2);
	
	real n3 = n1*n2-dot(v1,v2);
	real3 v3 = n1*v2 + n2*v1 + cross(v1,v2);
	return make_quaternion(n3,v3);
      }

      __host__ __device__ real3 getV1(real4 q){
	//Returns the first vector of the reference system encoded by the quaternion
	real a = q.x;
	real b = q.y;
	real c = q.z;
	real d = q.w;
	return make_real3(a*a+b*b-c*c-d*d,2*(b*c+a*d),2*(b*d-a*c));
      }
      
      __host__ __device__ real3 getV2(real4 q){
	//Returns the second vector of the reference system encoded by the quaternion
	real a = q.x;
	real b = q.y;
	real c = q.z;
	real d = q.w;
	return make_real3(2*(b*c-a*d),a*a-b*b+c*c-d*d,2*(c*d+a*b));
      }
      __host__ __device__ real3 getV3(real4 q){
	//Returns the third vector of the reference system encoded by the quaternion
	real a = q.x;
	real b = q.y;
	real c = q.z;
	real d = q.w;
	return make_real3(2*(b*d+a*c),2*(c*d-a*b),a*a-b*b-c*c+d*d);  
      }
      
      std::vector<real3> getReferenceSystem(real4 q){
	//Returns the basis of the reference system encoded by the quaternion
	real a = q.x;
	real b = q.y;
	real c = q.z;
	real d = q.w;
	std::vector<real3> basis(3);
	basis[0] = make_real3(a*a+b*b-c*c-d*d,2*(b*c+a*d),2*(b*d-a*c));
	basis[1] = make_real3(2*(b*c-a*d),a*a-b*b+c*c-d*d,2*(c*d+a*b));
	basis[2] = make_real3(2*(b*d+a*c),2*(c*d-a*b),a*a-b*b-c*c+d*d);  
	return basis;
      }
      
      std::vector<real4> initOrientations(int nParticles, uint seed, std::string type){
	//Set the initial orientations of each particle
	std::vector<real4> orientations(nParticles);
	if (type=="aligned"){
	  //All the particles are aligned with the laboratory frame
	  std::fill(orientations.begin(), orientations.end(),make_real4(1,0,0,0));
	} else if (type=="random"){
	  // The quaternions are generated randomly uniformly distributed
	  // http://refbase.cvc.uab.es/files/PIE2012.pdf
	  Saru rng(seed);
	  auto randomQuaternion = [&] (){
	    real x0 = rng.f();
	    real r1 = sqrt(1.0-x0);
	    real r2 = sqrt(x0);
	    real ang1 = 2*M_PI*rng.f();
	    real ang2 = 2*M_PI*rng.f();
	    return make_real4(r2*cos(ang2),r1*sin(ang1),r1*cos(ang1),r2*sin(ang2));
	  };
	  std::generate(orientations.begin(), orientations.end(),randomQuaternion);
	}
	return orientations;
      }      

      std::vector<real4> initOrientations(int nParticles, std::string type){
	uint seed = std::time(NULL);
	return initOrientations(nParticles, seed, type);
    }
  }
}
