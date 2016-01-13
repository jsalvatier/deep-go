AdagradOptimizer = {}
AdagradOptimizer.__index = AdagradOptimizer
function AdagradOptimizer.new(rate, decay)
    local self = setmetatable({}, AdagradOptimizer)
    self.grad_ms = grads:clone():fill(1)
    self.decay = decay
    self.rate = rate
    return self
end
function AdagradOptimizer:step(parameters, grad)
    self.grad_ms:mul(self.decay)
    self.grad_ms = self.grad_ms:add(1 - self.decay, torch.cmul(grad,grad))
    parameters:add(-self.rate, torch.cdiv(grad, torch.sqrt(self.grad_ms)))
end

SGD = {}
SGD.__index = SGD
function SGD.new(rate, decay)
    local self = setmetatable({}, SGD)
    self.rate = rate
    self.rate_decay = decay
    return self
end
function SGD:step(parameters, grad)
    parameters:add(-self.rate, grad)
    self.rate = self.rate * (1 - self.rate_decay)
end
