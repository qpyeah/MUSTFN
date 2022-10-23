# MFSTFN
Remote sensing image spatiotemporal fusion algorithm
Remote sensing image spatialtemporal fusion algorithm for MODIS generation of Landsat-7, Landsat-8 and GF-1


input :
F: landsat
M: modis


label_mask = F2 + MASK(cloud:1, nocloud:0) , total 7bands


landsat_modis = train = 6bands F1 + 6bands F2 + 7bands M1 + 7bands M2 + 7 bands M3



input 

F：GF-1 4 bands
M：MODIS  bands


