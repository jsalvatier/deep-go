require 'experiments'
require 'cutorch'

experiment = basicGoExperiment:new { 
    name = "default experiment",
    dataset = GoDataset,
    group = "train",
    useCuda = true,
    validationSize = 1000,
    numLayers = 3,
    channelSize = 64,

    batchSize = 64,
    kernels = {5},
    strides = {1},
    channels = {37},

    
    rate = .128,
    rateDecay = 1e-7,

    criterion = nn.ClassNLLCriterion()
}

print(cutorch.setDevice(1))


experiment:init()
print(experiment.id)

experiment:run{iters=200}
--experiment:run{iters=1000000}
