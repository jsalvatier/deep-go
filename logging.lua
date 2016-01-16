require 'io'

function log(experiment, trainCost, runningTime)
  local last_train = 'nope'
  if #trainCost ~= 0 then
    last_train = trainCost[#trainCost]
  end
  local last_val = 'nope'
  if #experiment.validation_costs ~= 0 then
    last_val = experiment.validation_costs[#experiment.validation_costs]
  end

  io.popen("curl " ..
      "-F entry.1216864853='" .. experiment.name .. ":" .. experiment.id .. "' " ..
      "-F entry.1905150392='" .. experiment.numLayers .. "' " ..
      "-F entry.1937596538='" .. experiment.channelSize .. "' " ..
      "-F entry.1622495815='" .. experiment.batchSize .. "' " ..
      "-F entry.481250734='" .. experiment.rate .. "' " ..
      "-F entry.686899840='" .. experiment.rateDecay .. "' " ..
      "-F entry.1314668940='" .. last_train .. "' " ..
      "-F entry.150702719='" .. runningTime .. "' " ..
      "-F entry.1128574508='" .. last_val .. "' " ..
      "-F entry.814635649='" .. experiment.iterations .. "' " ..
      "https://docs.google.com/forms/d/1_Ef8l0FistzfuQFS13SNthQfztbx-24L9AKFZvQc54s/formResponse")
end
