require 'data'
require 'torch'

GoDataset = Dataset:new()
GoDataset.output_dimensions = {}

SIZE = 19

--definitions of channels
--all features are from the perspective of black
STONE=1 --1=black, 2=white, 3=empty
LIBERTIES=STONE+3 --1 liberty, 2 liberties, ..., 4 or more liberties
LIBERTIES_AFTER=LIBERTIES+4 -- 0, 1, 2, ..., 6 or more
KILL=LIBERTIES_AFTER+7 --1, 2, ..., 7 or more
AGE=KILL+7 --1, 2, ..., 5
LADDER=AGE+5
RANK=LADDER+1 --1d, 2d, ..., 9d
TOTAL = RANK+9

INPUT = {}
INPUT.stone = 1
INPUT.liberties = INPUT.stone+1
INPUT.liberties_after = INPUT.liberties+1
INPUT.kills = INPUT.liberties_after+2
INPUT.age = INPUT.kills+2
INPUT.ladder = INPUT.age+1
INPUT.total = INPUT.ladder+1

-- this is used by makedata.lua when writing data
function flatten_data(data)
    local result = torch.ByteTensor(INPUT.total, SIZE, SIZE)
    result[INPUT.stone] = data.stones
    result[INPUT.liberties] = data.liberties
    result[{{INPUT.liberties_after, INPUT.liberties_after+1}}] = data.liberties_after or data.liberties_after_move
    result[{{INPUT.kills, INPUT.kills+1}}] = data.kills
    result[INPUT.age] = data.age or data.ages
    result[{{INPUT.ladder, INPUT.ladder+1}}] = data.ladders or torch.zeros(2, SIZE, SIZE)
    return {input=result, move=data.move, ranks=data.ranks, flat=true}
end


GoDataset.input_dimensions = {TOTAL, SIZE, SIZE}

function transform(grid)
    -- eventually this should do random rotation and reflection, but for now lets just skip it
    return grid
end

function expand_to(n, x)
    return x:repeatTensor(n, 1, 1)
end

function GoDataset:preprocess(data)
    local input = torch.DoubleTensor(torch.LongStorage(GoDataset.input_dimensions)):zero()
    local raw = data.input
    raw = transform(raw)
    local player = data.move.player


    input[STONE] = torch.eq(raw[INPUT.stone], 0)
    input[STONE+1] = torch.eq(raw[INPUT.stone], player)
    input[STONE+2] = torch.eq(raw[INPUT.stone], 3-player)


    local numbers = torch.ByteTensor(7, SIZE, SIZE)
    for i = 1, 7 do
        numbers[i] = torch.ones(SIZE, SIZE) * i
    end

    local raw_liberties = raw[INPUT.liberties]
    input[{{LIBERTIES, LIBERTIES+2}}][{{1, 3}}] = torch.eq(expand_to(3,raw_liberties), numbers[{{1, 3}}])
    input[LIBERTIES+3] = torch.ge(raw_liberties, 4)


    local raw_liberties_after = raw[INPUT.liberties_after + player - 1]
    input[LIBERTIES_AFTER]:addcmul(input[STONE], torch.eq(raw_liberties_after, 0):double())
    input[{{LIBERTIES_AFTER+1, LIBERTIES_AFTER+5}}] = torch.eq(expand_to(5,raw_liberties_after), numbers[{{1, 5}}])
    input[LIBERTIES_AFTER+6] = torch.ge(raw_liberties_after, 6)

    local raw_kills = raw[INPUT.kills + player - 1]
    input[{{KILL, KILL+5}}] = torch.eq(expand_to(6,raw_kills), numbers[{{1,6}}])
    input[KILL+6] = torch.ge(raw_kills, 7)

    local raw_age = raw[INPUT.age]
    input[{{AGE, AGE+4}}] = torch.eq(expand_to(5,raw_age), numbers[{{1, 5}}])

    local raw_ladder = raw[INPUT.ladder + player - 1]
    input[LADDER] = torch.ge(raw_ladder, 1)

    input[RANK+data.ranks[player]] = numbers[1]

    local output = SIZE * (data.move.x - 1) + data.move.y
    return {input=input, output=output}

end

ToyDataset = GoDataset:new()

ToyDataset.directories = {toy='toy', validate='toy'}
ToyDataset.root = "data"
