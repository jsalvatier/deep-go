require 'experiments'
require 'os'

dataset = GoDataset:new{root="data"}
experiment = basicGoExperiment:new{useCuda=false, dataset=dataset}

experiment:run{iters=100}

