require 'experiments'
require 'os'

experiment = basicGoExperiment:new{
    useCuda=false,
    data_root="data",
    validation_interval=20,
    validation_size=100,
    batchSize = 2
}
experiment:run{iters=20}
