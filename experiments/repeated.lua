require 'experiments'
require 'cutorch'
require 'cunn'
require 'optimizer'

cmd = torch.CmdLine()
cmd:option('-gpu', 1, "gpu to run on")
cmd:option('-num', 1, "experiment number")
cmd:option('-iters', 100, "experiment number")
local opt = cmd:parse(arg or {})

cutorch.setDevice(opt.gpu)

local best_previous = '0.95223742560484.model'
experiment = Experiment:load(best_previous)
experiment.id = "repeatedRun" .. opt.num
experiment.optimizer = SGD.new(experiment.rate, experiment.rateDecay)


print("experiment rate", experiment.rate)

experiment:run{iters=opt.iters}
