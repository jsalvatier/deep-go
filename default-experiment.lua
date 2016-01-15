require 'experiments'
require 'cutorch'

local GPU = 1 
print('setting device to', GPU)
cutorch.setDevice(GPU)

experiment = basicGoExperiment:new { 
    name = "6 layers, rate=.512",
    dataset = GoDataset,
    group = "train",
    useCuda = true,
    validationSize = 1000,
    validation_interval = 20,
    
    numLayers = 6,
    channelSize = 64,

    batchSize = 64,
    kernels = {5},
    strides = {1},
    channels = {37},

    
    rate = .512,
    rateDecay = 1e-7,

    criterion = nn.ClassNLLCriterion()
}

experiment:init()
print('experiment id: ', experiment.id)

