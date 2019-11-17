# Panel Method 3D
This is a 3D air flow analysis program, which uses panel method based on triangular mesh for calculation.

For now there is only a slovenian version of the program.

How to use:

  1. Copy all of the files in a single directory (or adapt all the dependancies).
  2. Open "program4.py".
  3. Under "STL MODEL" choose your mesh location, then choose the angle of attack "kot_alfa". 
  4. The program consists of 11 steps, that need to be run in order to perform calculation. Run them consequentaly an use parameter "koraki" to determin which steps to perform in one cycle.
  5. All the calculations and matrices will be stored in a new directory called "vmesni_rezultati".

Example on a sphere: tangential velocity color map (visualization done with Paraview)

<p align="center">
  <img src="https://i.imgur.com/IOXQoB4.png" width="350" title="Demo sphere">
</p>
