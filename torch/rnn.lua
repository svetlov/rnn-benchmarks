require('torch')
require('cutorch')
require('nn')
require('cunn')
require('rnn')
require('nngraph')

-- Should produce a speed increase.
nn.FastLSTM.usenngraph = true

-- cutorch.setDevice(2)

cmd = torch.CmdLine()
cmd:text()
cmd:text('Options')
cmd:option('-nbatch', 1000, 'Number of samples')
cmd:option('-network', 'fastlstm', 'Network type')
cmd:option('-hiddensize', 128, 'Neural network input and output size')
cmd:option('-seqlen', 30, 'Sequence length')
cmd:option('-batchsize', 20, 'Batch size')
cmd:text()
for k, v in pairs(cmd:parse(arg)) do _G[k] = v end

local input = torch.rand(seqlen, batchsize, hiddensize):cuda()
local target = torch.rand(batchsize, hiddensize):cuda()

local a = torch.Timer()
local rnn
if network == 'rnn' then
   rnn = nn.Sequential()
      :add(nn.JoinTable(1,1))
      :add(nn.Linear(hiddensize*2, hiddensize))
      :add(nn.Sigmoid())
   rnn = nn.Recurrence(rnn, hiddensize, 1)
   rnn = nn.Sequential()
      :add(nn.Sequencer(rnn))
      :add(nn.Select(1,-1))
elseif network == 'lstm' then -- ( no peephole connections)
   rnn = nn.LSTM(hiddensize, hiddensize)
   rnn = nn.Sequential()
      :add(nn.Sequencer(rnn))
      :add(nn.Select(1,-1))
elseif network == 'oldfastlstm' then -- ( no peephole connections)
   rnn = nn.FastLSTM(hiddensize, hiddensize)
   rnn = nn.Sequential()
      :add(nn.Sequencer(rnn))
      :add(nn.Select(1,-1))
elseif network == 'fastlstm' then -- like fastlstm but faster ( no peephole connections)
   rnn = nn.SeqLSTM(hiddensize, hiddensize)
   rnn = nn.Sequential()
      :add(rnn)
      :add(nn.Select(1,-1))
else
   error('Unkown network type!')
end

local criterion = nn.MSECriterion()
if cpu ~= true then
   rnn:cuda()
   criterion:cuda()
end

criterion:forward(rnn:forward(input), target)
rnn:backward(input, criterion:backward(rnn.output, target))
cutorch.synchronize()
print("Setup : compile + forward/backward x 1")
print("--- " .. a:time().real .. " seconds ---")

a:reset()
for i = 1, nbatch do
   rnn:forward(input)
end
cutorch.synchronize()
print("Forward:")
local nSamples = nbatch * batchsize
local speed = nSamples / a:time().real
print("--- " .. nSamples .. " samples in " .. a:time().real .. " seconds (" .. speed .. " samples/s, " .. 1000000/speed .. " microsec/samples) ---")

a:reset()
for i = 1, nbatch do
   criterion:forward(rnn:forward(input), target)
   rnn:zeroGradParameters()
   rnn:backward(input, criterion:backward(rnn.output, target))
   rnn:updateParameters(0.01)
end
cutorch.synchronize()
print("Forward + Backward:")
local speed = nSamples / a:time().real
print("--- " .. nSamples .. " samples in " .. a:time().real .. " seconds (" .. speed .. " samples/s, " .. 1000000/speed .. " microsec/samples) ---")
