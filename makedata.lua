require 'io'
require 'os'
require 'paths'

-- sgf encodes indices as chars; this table converts them back
chars_to_ints = {}
chars = "abcdefghijklmnopqrs"
assert(#chars == 19)
for i = 1, 19 do
    chars_to_ints[chars:sub(i, i)] = i
end

function string:split(pat)
  pat = pat or '%s+'
  local st, g = 1, self:gmatch("()("..pat..")")
  local function getter(segs, seps, sep, cap1, ...)
    st = sep and seps + #sep
    return self:sub(segs, (seps or 0) - 1), cap1 or sep, ...
  end
  return function() if st then return getter(st, g()) end end
end

function handicaps(s)
    local result = {}
    for line in s:split("\r\n") do
        if line:sub(1, 2) == "AB" or line:sub(1, 2) == "AW" then
            local player
            if line:sub(1, 2) == "AW" then player = 2 else player = 1 end
            for pos in line:sub(4,-2):split("%]%[") do
                local move = {player=player}
                move.x, move.y = to_move(pos)
                table.insert(result, move)
            end
        end
    end
    return result
end

function split_sgf(s)
    -- splits an SGF file into a sequence of (variable, value) tokens
    lines = s:split("\r\n")
    pieces = {}
    for line in lines do
        for piece in line:split(";") do
            table.insert(pieces, piece)
        end
    end
    result = {}
    for i, piece in pairs(pieces) do
        local subpieces = {}
        for subpiece in piece:split("%[") do table.insert(subpieces, subpiece) end
        if #subpieces == 2 and subpieces[2]:sub(-1) == "]" then
            table.insert(result, {subpieces[1], subpieces[2]:sub(1,-2)})
        end
    end
    return result
end

function to_move(s)
    -- turns a length 2 string s, in sgf format, into a pair of integers
    if #s == 2 then
        return chars_to_ints[s:sub(1,1)], chars_to_ints[s:sub(2,2)]
    else
        return nil
    end
end

function all_moves(s)
    -- returns an iterator that returns a sequence of moves
    -- each move is a table with player, x, y values
    -- s is a string in sgf format
    local entries = split_sgf(s)
    local i = 1
    local function iter()
        if i > #entries then return nil end
        local entry = entries[i]
        i = i+1
        if entry[1] == "B" then
            x, y = to_move(entry[2])
            -- x is nil if one player passes...
            if x ~= nil then return {player=1, x=x, y=y} end
        elseif entry[1] == "W" then
            x, y = to_move(entry[2])
            if x ~= nil then return {player=2, x=x, y=y} end
        end
        return iter()
    end
    return iter
end

function to_rank(s)
    -- turns a string s into a rank
    -- returns nil if s does not represent a dan rank
    if s:sub(-1) == "d" then
        return tonumber(s:sub(1, -2))
    else
        return nil
    end
end

function get_ranks(s)
    -- returns a pair (rank of player 1, rank of player 2)
    -- s is a string in sgf format
    local entries = split_sgf(s)
    result = {}
    for i = 1, #entries do
        entry = entries[i]
        if entry[1] == "BR" then
            result[1] = to_rank(entry[2])
        elseif entry[1] == "WR" then
            result[2] = to_rank(entry[2])
        end
    end
    if result[1] ~= nil and result[2] ~= nil then
        return result
    else
        return nil
    end
end

function all_kills_and_liberties_after(board)
    local kills, liberties_after = {}, {}
    for k = 1, 2 do
        kills[k] = {}
        liberties_after[k] = {}
        for i = 1, 19 do
            liberties_after[k][i] = {}
            kills[k][i] = {}
            for j = 1,19 do
                if board[i][j] == 0 then
                    kills[k][i][j], liberties_after[k][i][j] = count_kills_and_liberties(board, i, j, k)
                else
                    liberties_after[k][i][j] = 0
                    kills[k][i][j] = 0
                end
            end
        end
    end
    return kills, liberties_after
end

function summarize_board(stones)
    local kills, liberties_after = all_kills_and_liberties_after(stones)
    local ladders, liberties = all_ladder_moves_and_liberties(stones)
    return {
        stones=torch.ByteTensor(stones),
        liberties=torch.ByteTensor(liberties),
        liberties_after=torch.ByteTensor(liberties_after),
        ladders=torch.ByteTensor(ladders),
        kills=torch.ByteTensor(kills)
    }
end


function all_boards(handicaps, moves)
    -- returns an iterator that returns sequence of tables with board and move fields
    -- board is encoded as two 19x19 grids
    --   the first one is 0, 1, 2 according to whether player 1 or 2 has a piece there
    --   the second is an integer encoding how many turns a square has had the same state
    -- each move is encoded as a table with player, x, y fields
    -- moves is an iterator that returns a sequence of moves
    -- handicaps is a list of moves that should be played before the game begins
    local stones, age = {}, {}
    for i = 1, 19 do
        stones[i] = {}
        age[i] = {}
        for j = 1, 19 do
            stones[i][j] = 0
            age[i][j] = 0
        end
    end
    for _, handicap in pairs(handicaps) do
        update_board(stones, age, handicap)
    end
    local function iter()
        local move = moves()
        if move == nil then return nil end
        local result = summarize_board(stones)
        result.move = move
        result.age = torch.ByteTensor(age)
        update_board(stones, age, move)
        return result
    end
    return iter
end

function hash(x, y)
    -- converts a board position into an integer in [0, 390]
    return 19*(x-1) + y - 1
end

function unhash(z)
    -- converts an integer in [0, 390] into a board position
    return math.floor(z/19)+1, math.floor(z) % 19 +1
end

-- the list of displacements to adjacent squares
directions = {{-1, 0}, {1, 0}, {0, -1},{0, 1}}

function is_valid(a, b)
    -- returns whether a, b is a valid square on aboard
    return 1 <= a and a <= 19 and 1<= b and b <= 19
end


function neighbors(a, b)
    -- returns an iterator that enumerates all pairs na, nb adjacent to a, b on the board
    local i = 1
    local function iter()
        if i > #directions then return nil end
        dir = directions[i]
        i = i + 1
        na, nb= a + dir[1], b + dir[2]
        if is_valid(na, nb) then
            return na,nb 
        else
            return iter()
        end
    end
    return iter
end

function apply_f_if_dead(board, x, y, f)
    local liberties, group = count_liberties(board, x, y, true)
    if liberties == 0 and board[x][y] > 0 then
        for h, _ in pairs(group) do
            local a, b = unhash(h)
            f(a, b, 0)
        end
    end
end

function apply_f_to_dead_neighbors(board, x, y, f)
    for a, b in neighbors(x, y) do
        if board[a][b] == 3 - board[x][y] then
            apply_f_if_dead(board, a, b, f)
        end
    end
    apply_f_if_dead(board, x, y, f)
end



function count_liberties(board, x, y, count)
    -- counts the liberties at the group containing (x, y)
    -- if player is not 0, then count liberties after player plays there
    --
    if count == nil then count = true end
    local group = {}
    local frontier = {}
    local liberties = {}
    local player = board[x][y]
    if player == 0 then return 0 end

    frontier[hash(x,y)] = 1

    while next(frontier) ~= nil do
        local h = next(frontier)
        frontier[h] = nil
        group[h] = 1
        local a, b = unhash(h)
        for na, nb in neighbors(a, b) do
            local nh = hash(na, nb)
            if board[na][nb] == player then
                if group[nh] == nil then
                    frontier[nh] = 1
                end
            elseif board[na][nb] == 0 and group[nh] ~= 1 then
                if not count then
                    return 1
                end
                liberties[nh] = 1
            end
        end
    end

    local result = 0
    for _, _ in pairs(liberties) do result = result + 1 end

    return result, group, liberties
end

function equal_boards(a, b)
    for i = 1, 19 do
        for j = 1, 19 do
            if a[i][j] ~= b[i][j] then return false, {i,j} end
        end
    end
    return true
end

function copy_board(b)
    local result = {}
    for i = 1, 19 do
        result[i] = {}
        for j = 1, 19 do
            result[i][j] = b[i][j]
        end
    end
    return result
end

function count_kills_and_liberties(board, x, y, player)
    local kills = 0
    local to_unwind = {}
    local function temp_play_and_count(i, j, p)
        if board[i][j] ~= 0 and p ~= 0 then
            error("imagining playing in a place where we have already played")
        end
        if p == 0 and board[i][j] == 3 - player then
            kills = kills + 1
        end
        table.insert(to_unwind, {i,j,board[i][j]})
        board[i][j] = p
    end
    local function unwind()
        for i = #to_unwind, 1, -1 do
            m = to_unwind[i]
            board[m[1]][m[2]] = m[3]
        end
    end
    play_with_f(board, x, y, player, temp_play_and_count)
    local liberties = count_liberties(board, x, y, true)
    unwind()
    return kills, liberties
end

function update_board(stones, age, move)
    -- updates stones and age in place, given that move was made
    if age ~= nil then
        for i = 1, 19 do
            for j = 1, 19 do
                if age[i][j] > 0 and age[i][j] < 255 then
                    age[i][j] = age[i][j] + 1
                end
            end
        end
    end
    local x, y, player = move.x, move.y, move.player

    local function set(i, j, p)
        if age ~= nil then
            age[i][j] = 1
        end
        stones[i][j] = p
    end
    local function get(i,j)
        return stones[i][j]
    end

    if stones[x][y] ~= 0 then error("playing somewhere that's already been played!") end
    play_with_f(stones, x, y, player, set)
end

function make_dir(path)
    local prefix = ""
    for part in path:split("/") do
        prefix = prefix..part.."/"
        os.execute("mkdir "..prefix)
    end
end

function targets_for(source, sourcedir, targetdir)
    local path = translate_from_to(source, sourcedir, targetdir)
    if paths.filep(paths.concat(path, "100")) then return nil end
    make_dir(path)
    i = 0
    local function iter()
        i = i + 1
        return path.."/"..i
    end
    return iter
end

function transcribe_from(source, sourcedir, targetdir)
    print("transcribing...", source)
    local targets = targets_for(source, sourcedir, targetdir)
    if targets ~= nil then
        transcribe_from_to(source, targets_for(source, sourcedir, targetdir))
    end
end

function transcribe_from_list(sources, sourcedir, targetdir)
    for source in sources do transcribe_from(source, sourcedir, targetdir) end
end

function play_with_f(board, x, y, player, f)
    f(x, y, player)
    apply_f_to_dead_neighbors(board, x, y, f)
end

function ladder_moves(board, x, y, liberty_set)
    local to_unwind = {}
    local function temp_play(i, j, p)
        if board[i][j] ~= 0 and p ~= 0 then
            error("imagining playing in a place where we have already played")
        end
        table.insert(to_unwind, {i,j,board[i][j]})
        board[i][j] = p
    end
    local function unwind()
        for i = #to_unwind, 1, -1 do
            m = to_unwind[i]
            board[m[1]][m[2]] = m[3]
        end
        to_unwind = {}
    end
    local player = board[x][y]
    local opp = 3 - player
    local result = {}

    local liberties = {}
    for h, _ in pairs(liberty_set) do
        local a, b = unhash(h)
        table.insert(liberties, {a, b})
    end
    assert(#liberties == 2)
    for i = 1, 2 do
        local a, b = liberties[i][1], liberties[i][2]
        local oa, ob = liberties[3-i][1], liberties[3-i][2]
        play_with_f(board, a, b, opp, temp_play)
        local n = count_liberties(board, a, b)
        if n > 2 then
            play_with_f(board, oa, ob, player, temp_play)
            local n, _, new_liberties = count_liberties(board, oa, ob)
            if n == 1 then
                table.insert(result, {a, b})
            elseif n == 2 then
                n = count_liberties(board, a, b)
                if n > 1 and #ladder_moves(board, x, y, new_liberties) > 0 then
                    table.insert(result, {a, b})
                end
            end
        end
        unwind()
    end
    return result
end

function all_ladder_moves_and_liberties(board)
    local ladders, liberties = {}, {}
    for k = 1, 2 do
        ladders[k] = {}
    end
    for i = 1, 19 do
        for k = 1, 2 do
            ladders[k][i] = {}
        end
        liberties[i] = {}
        for j = 1, 19 do
            for k = 1, 2 do
                ladders[k][i][j] = 0
            end
            liberties[i][j] = 0
        end
    end
    local considered = {}
    for i = 1, 19 do
        for j = 1, 19 do
            if board[i][j] ~= 0 and considered[hash(i, j)] ~= 1 then
                local num_liberties, group, liberty_list = count_liberties(board, i, j)
                local groupsize = 0
                for h, _ in pairs(group) do
                    considered[h] = 1
                    groupsize = groupsize + 1
                    local a, b = unhash(h)
                    liberties[a][b] = num_liberties
                end
                if num_liberties == 2 then
                    for _, ladder_move in pairs(ladder_moves(board, i, j, liberty_list)) do
                        ladders[3-board[i][j]][ladder_move[1]][ladder_move[2]] = groupsize
                    end
                end
            end
        end
    end
    return ladders, liberties
end




function transcribe_from_file(filename, sourcedir, targetdir)
    local f = io.open(filename)
    transcribe_from_list(f:lines(), sourcedir, targetdir)
    f:close()
end

function all_files(dir)
    dir = dir or "data/raw"
    local result = {}
    for line in io.popen("find "..dir.." -type f"):lines() do
        table.insert(result, line)
    end
    return result
end

function shuffle(l)
    for i = 1, #l do
        local j = math.random(i, #l)
        l[i], l[j] = l[j], l[i]
    end
end

function split_to_groups(files, num_groups)
    local n = #files
    for i = 1, num_groups do
        local f = io.open("transcription_group_"..i, "w")
        for j = math.floor((i-1)*n/num_groups) + 1, math.floor(i*n/num_groups) do
            f:write(files[j].."\n")
        end
        f:close()
    end
end

function transcribe_from_to(source, targets)
    -- source is the filename from which to read a Go game, in sgf format
    -- targets is an iterator that generates filenames to which to write data points
    -- the outputs are serialized torch objects 

    local f = io.open(source, "r")
    local s = f:read("*all")

    local moves = all_moves(s)
    local handicaps = handicaps(s)
    local ranks = get_ranks(s)
    local boards = all_boards(handicaps, moves)

    if ranks ~= nil then
        for board in boards do
            board.ranks = ranks
            local target = targets()
            torch.save(target, board)
        end
    end
    
    f:close()
end

function dumb_target()
    local function iter()
        return "test"
    end
    return iter
end

function transcribe_all(groups)
    local sources = all_sources()
    for group, size in pairs(groups) do
        local targets = all_targets(group)
        for i = 1, size do
            local source = sources()
            print("transcribing...", source, group..i)
            transcribe(source, targets)
        end
    end
end

function scatter_to_categories(categories, sourcedir, targetdir)
    local files = all_files(sourcedir)
    shuffle(files)
    i = 0
    paths.mkdir(targetdir)
    for category, size in pairs(categories) do
        paths.mkdir(targetdir.."/"..category)
        for s in paths.iterdirs(sourcedir) do
            paths.mkdir(targetdir.."/"..category.."/"..s)
        end
        for j = 1, size do
            i = i + 1
            if i > #files then return end
            local filename = files[i]
            local destination = translate_from_to(filename, sourcedir, targetdir.."/"..category)
            copy_file(filename, destination, false)
        end
    end
end

function translate_from_to(name, source, target)
    local n = #source
    if n == 0 then return target.."/"..name end
    return target..name:sub(n+1)
end

function copy_file(source, target, need_dir)
    if need_dir then make_dir(paths.dirname(target)) end
    os.execute("cp "..source.." "..target)
end

function all_sources()
    return io.popen("find raw -type f"):lines()
end

function all_targets(dirname)
    dirname = dirname or "processed"
    local branching = 1000
    local i,j,k = 0,0,0
    root = {}
    os.execute("mkdir "..dirname)
    local function mkdir(xs)
        local prefix = dirname
        local dir = root
        for _, x in pairs(xs) do
            prefix = prefix.."/"..x
            if dir[x] == nil then
                dir[x] = {}
                os.execute("mkdir "..prefix)
            end
            dir = dir[x]
        end
    end
    local function iter()
        mkdir({i,j})
        result = dirname.."/"..i.."/"..j.."/"..k
        k = k+1
        if k > branching then
            k = 0
            j = j+1
        end
        if j > branching then
            j = 0
            i = i+1
        end
        if k > branching then
            return nil
        end
        return result
    end
    return iter
end
