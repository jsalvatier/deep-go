require 'experiments'
require 'cutorch'
require 'cunn'

cmd = torch.CmdLine()
cmd:option('-gpu', 1, "gpu to run on")
cmd:option('-num', 1, "experiment number")
cmd:option('-iters', 100, "experiment number")
local opt = cmd:parse(arg or {})

local best_previous = '0.95223742560484.model'
experiment = torch.load(best_previous)
experiment.id = "repeatedRun" .. opt.num
cutorch.setDevice(opt.gpu)

print("experiment rate", experiment.rate)

experiment:run{iters=iters}
