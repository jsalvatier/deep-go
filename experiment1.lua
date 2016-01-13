require 'model'
require 'train'
require 'optimizer'
require 'logging'

useCuda = true
iterations = 100
numLayers = 12
channelSize = 64

batchSize = 32 
kernels = {5}
strides = {1}
channels = {36}

for i = 2,numLayers do
    table.insert(kernels, 3)
    table.insert(strides, 1)
    table.insert(channels, channelSize)
end
table.insert(channels, 1)


model = getBasicModel(numLayers, kernels, channels)
criterion = nn.ClassNLLCriterion()

rate = .01
rateDecay = 1e-7
optimizer = SGD.new(rate, rateDecay)

start_time = sys.clock()
train_cost = train(model, criterion, batchSize, iterations, optimizer, useCuda)
running_time = sys.clock() - start_time

log('basic experiment', numLayers, channelSize, batchSize, rate, rateDecay, train_cost, running_time)
