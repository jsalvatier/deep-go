function getBasicModel(numLayers, kernels, channels) 
        smodel = nn.Sequential()
	for layer = 1, numLayers do
	    local padding = (kernels[layer] - 1)/2
	    smodel:add(nn.SpatialZeroPadding(padding, padding, padding, padding))
	    smodel:add(nn.SpatialConvolutionMM(channels[layer], channels[layer+1], kernels[layer], kernels[layer]))
	    
	    d1 = channels[layer+1]
	    d2 = 19
	    d3 = 19
	    smodel:add(nn.Reshape(d1*d2*d3))
	    smodel:add(nn.Add(d1*d2*d3))
	    smodel:add(nn.Reshape(d1, d2, d3))
	    
	    smodel:add(nn.ReLU())
	end
	smodel:add(nn.Reshape(19*19))
	smodel:add(nn.LogSoftMax())
	return smodel
end
