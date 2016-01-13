Plot = require 'itorch.Plot'

function plotTrain(train_cost)
  Plot():line(torch.range(1,#train_cost,1), train_cost,'red','train_cost'):legend(true):title('Cost'):draw()
end

function plotTrainAndVal(train_cost, val_cost)
  Plot():line(torch.range(1,#train_cost,1), train_cost,'red','train_cost')
        :line(torch.range(1,#val_cost,1), val_cost, 'blue', 'val_cost')
      :legend(true):title('Cost'):draw()
end
