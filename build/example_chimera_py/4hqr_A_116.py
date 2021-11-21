from chimera import runCommand 
runCommand('open 0 ./build/4hqr_process.pdb') 
runCommand('select :116.A z<15') 
runCommand('write format pdb selected 0 ./build/example_pocket/4hqr_A_116_pocket.pdb') 
runCommand('close 0')