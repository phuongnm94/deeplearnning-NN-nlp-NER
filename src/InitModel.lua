require ('nn')
require ('rnn')
require('InitData')


---
-- Cai dat du lieu dau vao
--
-- @function [parent=#InitModel] GetInputEmbeddedLayer(rawDataInputSize, hiddenSize, mtWeightInit, bIsUseFeatures, rawFeatureInputSize)
-- @param rawDataInputSize so chieu vector du lieu dau vao = kich thuoc tu dien
-- @param hiddenSize so chieu tang an
-- @param mtWeightInit ma tran khoi tao word to vect
-- @param bIsUseFeatures su dung cac dac trung ngon ngu bo sung
-- @param rawFeatureInputSize so chieu cua vector dac trung ngon ngu
local function GetInputEmbeddedLayer(rawDataInputSize, hiddenSize, mtWeightInit, bIsUseFeatures, rawFeatureInputSize)

        local module = nil

        -- xu ly input data {wordVetor, featuresVector}
        local w2v = nn.LookupTable(rawDataInputSize, hiddenSize)
        if(mtWeightInit ~= nil) then
                w2v.weight = mtWeightInit
        end


        if(bIsUseFeatures == false) then
                module = w2v
        else
                -- cai dat ma tran features
                local feature = nn.Sequencer(nn.Linear(rawFeatureInputSize, hiddenSize))
                module = nn.Sequential()
                        :add(nn.ParallelTable():add(w2v):add(feature))
                        :add(nn.CAddTable())
        end

        ::_EXIT_FUNCTION::
        return module
end


---
-- Cai dat cau truc mang noron
--
-- @function [parent=#global] InitModelNN(sModelName, rawDataInputSize, hiddenSize, nIndex, mtWeightInit)
-- @param sModelName ten mang neron - cac loai mang co the cai dat = rnn/rnnLstm/brnnLstm
-- @param rawDataInputSize so chieu vector du lieu dau vao = kich thuoc tu dien
-- @param hiddenSize so chieu tang an
-- @param nIndex so nhan tu loai
-- @param mtWeightInit ma tran khoi tao word to vect
function InitModelNN(sModelName, rawDataInputSize, hiddenSize, nIndex, mtWeightInit, rawFeatureInputSize)

        local module = nil

        -- ---------------------------------------------------------------------------------------
        -- ---------------------------------------------------------------------------------------
        -- SETUP NERON NET
        -- ---------------------------------------------------------------------------------------
        -- ---------------------------------------------------------------------------------------

        local inputLayer = GetInputEmbeddedLayer(rawDataInputSize,hiddenSize,mtWeightInit,g_isUseFeatureWord,rawFeatureInputSize)

        -- mang rnn co ban
        if(sModelName == 'rnn') then
                -- build simple recurrent neural network
                local r = nn.Recurrent(
                        hiddenSize,
                        (
                        nn.Sequential()
                                : add(nn.LookupTable(rawDataInputSize, hiddenSize))
                                : add(nn.Add(hiddenSize))
                        ),
                        nn.Linear(hiddenSize, hiddenSize),
                        nn.Tanh(),
                        rho
                )

                local rnn = nn.Sequential()
                        :add(r)
                        :add(nn.Linear(hiddenSize, nIndex))
                        :add(nn.LogSoftMax())

                -- internally, rnn will be wrapped into a Recursor to make it an AbstractRecurrent instance.
                module = nn.Sequencer(rnn)
                --        rnn = nn.Recursor(rnn, rho)

                local moduleInitW2V = (module:get(1):get(1):get(1):get(2):get(1))
                print(moduleInitW2V.weight:size())

                if(mtWeightInit ~= nil) then
                        moduleInitW2V.weight = mtWeightInit
                end
                goto _EXIT_FUNCTION_
        end

        -- ---------------------------------------------------------------------------------------
        -- mang rnn - ket hop lstm
        if (sModelName == "rnnLstm") then

                local rnnLstm = nn.SeqLSTM(hiddenSize, nIndex)
                rnnLstm.batchfirst = true

--                local w2v = nn.LookupTable(rawDataInputSize, hiddenSize)
--                w2v.weight = mtWeightInit

                local sofmax = nn.Sequencer(nn.LogSoftMax())

                module = nn.Sequential()
                        :add(inputLayer)
                        :add(rnnLstm)
                        :add(sofmax)

                goto _EXIT_FUNCTION_
        end


        -- ---------------------------------------------------------------------------------------
        -- mang brnn - ket hop lstm
        if (module == nil or sModelName == "brnnLstm") then

                local brnn = nn.SeqBRNN(hiddenSize, nIndex, true)

                 -- Log sofmax
                local sofmax = nn.Sequencer(nn.LogSoftMax())

                module = nn.Sequential()
                        :add(inputLayer)
                        :add(brnn)
                        :add(sofmax)
                goto _EXIT_FUNCTION_

        end

        -- ---------------------------------------------------------------------------------------
        ::_EXIT_FUNCTION_::
        return module
end

function testModel()

        local hiddenSize =5
        local nIndex = 9
        local rawDataInputSize = 10
        local rawFeatureInputSize = 9

        local net = InitModelNN("brnnLstm-features",rawDataInputSize,hiddenSize,nIndex,mtWeightInit,rawFeatureInputSize)

        local inputLookupTbl=  torch.Tensor(2,7):apply(
                function ()
                        return torch.random(1,10)
                end
        )
        print(inputLookupTbl)

        local inputLinear = torch.Tensor(2,7,9):apply(function()
                return  (torch.random(100)% 2)
        end)
        print(inputLinear)

        local inputEmbeded = {inputLookupTbl, inputLinear}
        print(inputEmbeded)

        local outNet = net:forward(inputEmbeded)
        local _, outNetIdx = outNet:topk(1, true)

        print(outNet)
        print(outNetIdx)
end

--testModel()

