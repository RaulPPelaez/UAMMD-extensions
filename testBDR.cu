#include"uammd.cuh"
#include"BrownianRotation.cuh"
#include"utils/InitialConditions.cuh"


using std::make_shared;
using std::to_string;
using namespace uammd;
using namespace extensions;
real norm2(real3 v){
  return (dot(v,v));
}
real norm(real3 v){
  return sqrt(norm2(v));
}
   
void saveData(std::shared_ptr<uammd::ParticleData> pd, int N, real r, std::ofstream &file){
  using quaternion::getReferenceSystem;
  auto pos = pd -> getPos(access::location::cpu, access::mode::read);
  auto dir = pd -> getDir(access::location::cpu, access::mode::read);
  const int * index2id = pd->getIdOrderedIndices(access::location::cpu);
  auto save = [&](int id){
    real a = dir[id].x;
    real b = dir[id].y;
    real c = dir[id].z;
    real d = dir[id].w;
    real3 pi = make_real3(pos[id]);
    std::vector<real3> referenceSystem = getReferenceSystem(dir[id]);
    file<<pi<<" "<<r<<" 0\n";
    file<<pi+r*referenceSystem[0]<<" "<<r/5.0<<" 1\n";
    file<<pi+r*referenceSystem[1]<<" "<<r/5.0<<" 1\n";
    file<<pi+r*referenceSystem[2]<<" "<<r/5.0<<" 1\n";    
  };
  file<<"#\n";
  std::for_each(index2id,index2id+pos.size(),save);
}

void saveAngles(real4* dir, int nParticles, std::ofstream &file){
  /* The correlation of the angles must depend on time as [1]:
     <cos(theta_{t,t0})> = exp(-2*Dr*t)
     Where theta_{t,t0} is the angle between the orientation of the particle at
     time t0, and time t     
     [1] https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5453791/
   */
  
  file<<"#\n";
  fori(0,nParticles){
    real3 v3 = quaternion::getV3(dir[i]);    
    real cang = v3.z/norm(v3); // We initialize all the MNP aligned with the z axis
    if (cang>real(1.0)) cang = (real)1.0;
    else if (cang<real(-1.0)) cang = real (-1.0);
    file<<acos(cang)<<"\n";
  }
}

void saveMSD(real4 *pos,std::vector<real4> &init, int nParticles,std::ofstream &file){
  // In 3D the mean square displacement must depend on time as MSD = 6*Dt*t
  // All the particles must be initialized in the origin (0,0,0)
  file<<"# "<<"\n";
  fori(0,nParticles){
    real3 pi = make_real3(pos[i])-make_real3(init[i]);
    real msd = norm2(pi);
    file<<msd<<"\n";
  }
}

struct inputParams {
  int nParticles;
  int nsteps;
  int nsave;
  real dt;
  real r;
  real viscosity;
  real temperature;
};

#include"utils/InputFile.h"
inputParams readData(std::shared_ptr<System> sys, std::string file){

  inputParams par;
  InputFile in(file, sys);
  in.getOption("nParticles", InputFile::Required)>>par.nParticles;
  in.getOption("nsteps", InputFile::Required)>>par.nsteps;
  in.getOption("nsave", InputFile::Required)>>par.nsave;
  in.getOption("dt", InputFile::Required)>>par.dt;
  in.getOption("radius", InputFile::Required)>>par.r;
  in.getOption("viscosity", InputFile::Required)>>par.viscosity;
  in.getOption("temperature", InputFile::Required)>>par.temperature;
  par.viscosity/=(8*M_PI);
  return par;
}
   

int main(int argc, char* argv[]){

  auto sys = make_shared<System>(argc,argv);

  inputParams par = readData(sys,"Params.data");
  int nParticles = par.nParticles;
  real lbox = sqrt(nParticles)*2;
  Box box(lbox);
  uint seed = std::time(NULL);
  sys -> rng().setSeed(seed);
  auto pd = make_shared<ParticleData>(nParticles,sys);
  auto initPos = initLattice(box.boxSize, nParticles, sc);
  auto initDir = quaternion::initOrientations(nParticles,seed,"aligned");
  {
    auto pos = pd -> getPos(access::location::cpu, access::mode::write);
    auto dir = pd -> getDir(access::location::cpu, access::mode::write);
    std::copy(initPos.begin(), initPos.end(), pos.begin());
    std::copy(initDir.begin(), initDir.end(), dir.begin());
    
  }

  using integrator = BDR::BrownianRotation;
  integrator::Parameters brpar;
  brpar.dt = par.dt;
  brpar.viscosity = par.viscosity;
  brpar.hydrodynamicRadius = par.r;
  brpar.temperature = par.temperature;
  auto br = std::make_shared<integrator>(pd, sys, brpar);
  
  std::ofstream fileang("angulos.out");
  std::ofstream filepos("Pos.out");
  std::ofstream fileMSD("MSD.out");
  
  Timer tim;
  tim.tic();

  fori(0,par.nsteps){  
    if (i%par.nsave == 0){
      std::cout<<i<<std::endl;
      {
      auto dir = pd-> getDir(access::location::cpu, access::mode::read);
      auto pos = pd-> getPos(access::location::cpu, access::mode::read);
      saveMSD(pos.begin(), initPos, nParticles,fileMSD);
      saveAngles(dir.begin(), nParticles, fileang);
      }
      //saveData(pd, nParticles, r, filepos);
    }
    br->forwardTime();
  }
  std::cout<<tim.toc()<<"\n";
  return 0;
}
