require"data"
require"model"


--data
dataset = GoDataset

--model

--train
batch = dataset:minibatch("train", 32)


--evaluate
