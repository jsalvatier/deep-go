require 'experiments'
require 'os'

experiment = basicGoExperiment:new{useCuda=false, data_root="data"}
experiment:run{iters=100}

