Plot = require 'itorch.Plot'
require 'cutorch'
require 'nn'

function plotAll()
  for file in io.popen('ls ~/deep-go | grep .model'):lines() do
    print('plotting', file)
    plotFromFile(file)
  end
end


function plotFromFile(filename)
  loadedExperiment = torch.load(filename)
 print(loadedExperiment.validation_costs)
print(loadedExperiment.iterations)

  if #(loadedExperiment.validation_costs) <= 1 then
    print('not plotting because validation_costs is empty')
    return
  end

  plotVal(loadedExperiment.validation_costs, loadedExperiment.iterations)
end  

function plotVal(val_costs, iterations)
  local step = iterations / #val_costs
  Plot():line(torch.range(1,iterations,step), val_costs,'red','val_costs'):legend(true):title('Validation Cost'):draw()
end
