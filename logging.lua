require 'io'

function log(experiment, train_cost, runningTime)
  io.popen("curl -v " ..
      "-F entry.1216864853='" .. experiment.name .. "' " ..
      "-F entry.1905150392='" .. experiment.numLayers .. "' " ..
      "-F entry.1937596538='" .. experiment.channelSize .. "' " ..
      "-F entry.1622495815='" .. experiment.batchSize .. "' " ..
      "-F entry.481250734='" .. experiment.rate .. "' " ..
      "-F entry.686899840='" .. experiment.rateDecay .. "' " ..
      "-F entry.1314668940='" .. train_cost[#train_cost] .. "' " ..
      "-F entry.150702719='" .. runningTime .. "' " ..
      "https://docs.google.com/forms/d/1_Ef8l0FistzfuQFS13SNthQfztbx-24L9AKFZvQc54s/formResponse")
end
