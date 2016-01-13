require 'io'

function log(experimentName, numLayers, channelSize, batchSize, rate, 
             rateDecay, train_cost, runningTime)
  io.popen("curl -v " ..
      "-F entry.1216864853='" .. experimentName .. "' " ..
      "-F entry.1905150392='" .. numLayers .. "' " ..
      "-F entry.1937596538='" .. channelSize .. "' " ..
      "-F entry.1622495815='" .. batchSize .. "' " ..
      "-F entry.481250734='" .. rate .. "' " ..
      "-F entry.686899840='" .. rateDecay .. "' " ..
      "-F entry.1314668940='" .. train_cost[#train_cost] .. "' " ..
      "-F entry.150702719='" .. runningTime .. "' " ..
      "https://docs.google.com/forms/d/1_Ef8l0FistzfuQFS13SNthQfztbx-24L9AKFZvQc54s/formResponse")
end
