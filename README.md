# MFSTFN
Remote sensing image spatiotemporal fusion algorithm
Remote sensing image spatialtemporal fusion algorithm for MODIS generation of Landsat-7, Landsat-8 and GF-1


input :
F: landsat
M: modis
label_mask = F2 + MASK(cloud:1, nocloud:0) , total 5bands
landsat_modis = train = 4bands F1 + 4bands F2 + 4bands M1 + 4bands M2 + 4 bands M3, total 20bands
