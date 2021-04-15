## Community extensions for UAMMD  

This repository contains additions to the UAMMD framework: https://github.com/RaulPPelaez/UAMMD  

During your usage of UAMMD you might had to write some new interaction (like a bond potential) or something like a new Integrator module with new physics.  
If you wish to contribute to the project with these new additions you may do that in this repository.  

## How to contribute:  

Fork this repository and add your new functionality to the corresponding file (i.e. a new bond potential should be in BondPotentials.cuh).  
Submit your extension as a pull request.  

## Guidelines:  

Every object in this repository must be under the uammd::extensions namespace.  
Extensions should not add any external dependencies beyond the ones already available in UAMMD. If your extension depends on some other code please open an issue in order to discuss its addition.  
Regarding compilation, when cloning uammd, extensions (as in the files in this repository) will be placed under src/extensions if the project is cloned recursively.  
Document your code and add it to the wiki if you wish. Inclusion of tests (at least in the form of some reproducible result or sanity check) is also encouraged.  

## Example:  

See BondPotentials.cuh for an example extension.  


