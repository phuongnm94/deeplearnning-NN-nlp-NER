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
local function GetInputEmbeddedLayer2(rawDataInputSize, hiddenSize, mtWeightInit, bIsUseFeatures, rawFeatureInputSize)

        local module = nil

        -- xu ly input data {wordVetor, featuresVector}
        local w2v = nil
        if g_isUseMaskZeroPadding == true then
                w2v = nn.LookupTableMaskZero(rawDataInputSize, hiddenSize)
        else
                w2v = nn.LookupTable(rawDataInputSize, hiddenSize)
        end


        if(mtWeightInit ~= nil) then
                w2v.weight = mtWeightInit
        end

        if(bIsUseFeatures~= nil and bIsUseFeatures == true) then
                -- cai dat ma tran features
                --local feature = nn.Sequencer(nn.Linear(rawFeatureInputSize, hiddenSize))
                module = nn.Sequential()
                        :add(nn.ParallelTable():add(w2v):add(nn.CAddTable()))
                        :add(nn.JoinTable(3))
        else
                module = w2v
        end

        ::_EXIT_FUNCTION::
        return module
end

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
        local w2v = nil
        if g_isUseMaskZeroPadding == true then
                w2v = nn.LookupTableMaskZero(rawDataInputSize, hiddenSize)
        else
                w2v = nn.LookupTable(rawDataInputSize, hiddenSize)
        end


        if(mtWeightInit ~= nil) then
                w2v.weight = mtWeightInit
        end

        if(bIsUseFeatures~= nil and bIsUseFeatures == true) then
                -- cai dat ma tran features
                local feature = nn.Sequencer(nn.Linear(rawFeatureInputSize, hiddenSize))
                module = nn.Sequential()
                        :add(nn.ParallelTable():add(w2v):add(feature))
                        :add(nn.CAddTable())
        else
                module = w2v
        end

        ::_EXIT_FUNCTION::
        return module
end


---
-- Cai dat cau truc mang noron
--
-- @function [parent=#global] InitModelNN(sModelName, rawDataInputSize, hiddenSize, nCountLabel, mtWeightInit)
-- @param sModelName ten mang neron - cac loai mang co the cai dat = rnn/rnnLstm/brnnLstm
-- @param rawDataInputSize so chieu vector du lieu dau vao = kich thuoc tu dien
-- @param hiddenSize so chieu tang an
-- @param nCountLabel so nhan tu loai
-- @param mtWeightInit ma tran khoi tao word to vect
function InitModelNN(sModelName, rawDataInputSize, hiddenSize, nCountLabel, mtWeightInit, rawFeatureInputSize)

        local module = nil

        -- ---------------------------------------------------------------------------------------
        -- ---------------------------------------------------------------------------------------
        -- SETUP NERON NET
        -- ---------------------------------------------------------------------------------------
        -- ---------------------------------------------------------------------------------------

        local inputLayer = GetInputEmbeddedLayer(rawDataInputSize,hiddenSize,
                mtWeightInit,g_isUseFeatureWord,rawFeatureInputSize)
        
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
                        :add(nn.Linear(hiddenSize, nCountLabel))
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

                local rnnLstm, linear, softmax

                -- cai dat cac tang co ban
                -- init basic Layer of net
                rnnLstm = nn.SeqLSTM(hiddenSize, math.ceil(hiddenSize/2))
                rnnLstm.batchfirst = true
                linear = nn.Linear( math.ceil(hiddenSize/2), g_nCountLabel)
                softmax = nn.LogSoftMax()

                -- Cai dat sequencer va maskzero cho cac layer
                -- init option sequencer & maskzero
                if(g_isUseMaskZeroPadding~= nil and g_isUseMaskZeroPadding == true) then
                        rnnLstm.maskzero = true
                        linear = nn.Sequencer(nn.MaskZero(linear, 1))
                        softmax = nn.Sequencer(nn.MaskZero(softmax, 1))
                else
                        linear = nn.Sequencer(linear)
                        softmax = nn.Sequencer(softmax)
                end

                -- cai dat mang chinh
                -- init net
                module = nn.Sequential()
                        :add(inputLayer)
                        :add(rnnLstm)
                        :add(linear)
                        :add(softmax)

                goto _EXIT_FUNCTION_
        end


        -- ---------------------------------------------------------------------------------------
        -- mang brnn - ket hop lstm
        if (sModelName == "brnnLstm") then
                if g_isUseFeatureWord == false then
                        
                        local brnn, linear, softmax

                        -- cai dat cac tang co ban
                        -- init basic Layer of net
                        brnn = nn.SeqBRNN(hiddenSize, math.ceil(hiddenSize/2), true)
                        linear = nn.Linear( math.ceil(hiddenSize/2), g_nCountLabel)
                        softmax = nn.LogSoftMax()

                        -- Cai dat sequencer va maskzero cho cac layer
                        -- init option sequencer & maskzero
                        if(g_isUseMaskZeroPadding == true) then
                                brnn.forwardModule.maskzero = true
                                brnn.backwardModule.maskzero = true
                                linear = nn.Sequencer(nn.MaskZero(linear, 1))
                                softmax = nn.Sequencer(nn.MaskZero(softmax, 1))

                        else
                                linear = nn.Sequencer(linear)
                                softmax = nn.Sequencer(softmax)
                        end

                        -- cai dat mang chinh
                        -- init net
                        module = nn.Sequential()
                                :add(inputLayer)
                                :add(brnn)
                                :add(linear)
                                :add(softmax)
                       
                       goto _EXIT_FUNCTION_
                else
                        -- features co ban - add truc tiep vao tang dau tien
                        -- inputdims = hiddensize
                        if(g_iModelTest ==  0) then
                                
                                -- Bo sung linguistic features
                                local brnn, linear, softmax
                                
                                inputLayer = GetInputEmbeddedLayer(rawDataInputSize,hiddenSize,
                                        mtWeightInit,g_isUseFeatureWord,rawFeatureInputSize)
                                        
                                -- cai dat cac tang co ban
                                -- init basic Layer of net
                                brnn = nn.SeqBRNN(hiddenSize, math.ceil(hiddenSize/2), true)
                                linear = nn.Linear( math.ceil(hiddenSize/2), g_nCountLabel)
                                softmax = nn.LogSoftMax()

                                -- Cai dat sequencer va maskzero cho cac layer
                                -- init option sequencer & maskzero
                                if(g_isUseMaskZeroPadding == true) then
                                        brnn.forwardModule.maskzero = true
                                        brnn.backwardModule.maskzero = true
                                        linear = nn.Sequencer(nn.MaskZero(linear, 1))
                                        softmax = nn.Sequencer(nn.MaskZero(softmax, 1))

                                else
                                        linear = nn.Sequencer(linear)
                                        softmax = nn.Sequencer(softmax)
                                end

                                -- cai dat mang chinh
                                -- init net
                                module = nn.Sequential()
                                        :add(inputLayer)
                                        :add(brnn)
                                        :add(linear)
                                        :add(softmax)
                                goto _EXIT_FUNCTION_
                        end 
                        
                        -- features co ban - Join vao input tang dau tien 
                        -- inputdims = hiddensize + featuresDimes
                        if(g_iModelTest ==  1) then
                                -- Bo sung linguistic features
                                local brnn, linear, softmax

                                inputLayer = GetInputEmbeddedLayer2(rawDataInputSize,hiddenSize,
                                        mtWeightInit,g_isUseFeatureWord,rawFeatureInputSize)

                                -- cai dat cac tang co ban
                                -- init basic Layer of net
                                brnn = nn.SeqBRNN(hiddenSize+rawFeatureInputSize, math.ceil(hiddenSize/2), true)
                                linear = nn.Linear( math.ceil(hiddenSize/2), g_nCountLabel)
                                softmax = nn.LogSoftMax()

                                -- Cai dat sequencer va maskzero cho cac layer
                                -- init option sequencer & maskzero
                                if(g_isUseMaskZeroPadding == true) then
                                        brnn.forwardModule.maskzero = true
                                        brnn.backwardModule.maskzero = true
                                        linear = nn.Sequencer(nn.MaskZero(linear, 1))
                                        softmax = nn.Sequencer(nn.MaskZero(softmax, 1))

                                else
                                        linear = nn.Sequencer(linear)
                                        softmax = nn.Sequencer(softmax)
                                end

                                -- cai dat mang chinh
                                -- init net
                                module = nn.Sequential()
                                        :add(inputLayer)
                                        :add(brnn)
                                        :add(linear)
                                        :add(softmax)
                                goto _EXIT_FUNCTION_
                        end 
                        
                        -- features nang cao - Add vao input tang cuoi cung 
                        if(g_iModelTest ==  2) then
                        
                                -- Bo sung linguistic features
                                local brnn, linear, softmax

                                inputLayer = GetInputEmbeddedLayer(rawDataInputSize,hiddenSize,
                                        mtWeightInit,false,rawFeatureInputSize)

                                -- cai dat cac tang co ban
                                -- init basic Layer of net
                                brnn = nn.SeqBRNN(hiddenSize, math.ceil(hiddenSize/2), true)
                                
                                local brnnCover = nn.Sequential():add(inputLayer) :add(brnn)
                                local feature = nn.Sequencer(
                                                        nn.Sequential()
                                                                :add(nn.Tanh())
                                                                :add(nn.Linear(rawFeatureInputSize, hiddenSize/2))
                                                )
                                local inputCover = nn.Sequential() 
                                        :add(nn.ParallelTable(): add(brnnCover) :add(feature))
                                        :add(nn.CAddTable())
                                
                                linear = nn.Linear( math.ceil(hiddenSize/2), g_nCountLabel)
                                softmax = nn.LogSoftMax()

                                -- Cai dat sequencer va maskzero cho cac layer
                                -- init option sequencer & maskzero
                                if(g_isUseMaskZeroPadding == true) then
                                        brnn.forwardModule.maskzero = true
                                        brnn.backwardModule.maskzero = true
                                        linear = nn.Sequencer(nn.MaskZero(linear, 1))
                                        softmax = nn.Sequencer(nn.MaskZero(softmax, 1))

                                else
                                        linear = nn.Sequencer(linear)
                                        softmax = nn.Sequencer(softmax)
                                end

                                -- cai dat mang chinh
                                -- init net
                                module = nn.Sequential()
                                        --:add(inputLayer)
                                        --:add(brnn)
                                        :add(inputCover)
                                        :add(linear)
                                        :add(softmax)
                                        
                                goto _EXIT_FUNCTION_
                        end                        
                
                end
                
        end

        -- ---------------------------------------------------------------------------------------
        ::_EXIT_FUNCTION_::
        return module
end

function testModel()

        local hiddenSize =10
        local nCountLabel = 9
        local rawDataInputSize = 20
        local rawFeatureInputSize = 15
        w2v = nn.LookupTableMaskZero(rawDataInputSize, hiddenSize)
        g_isUseMaskZeroPadding =true
        g_isUseFeatureWord =true
        local net = InitModelNN("brnnLstm",rawDataInputSize,hiddenSize,nCountLabel,
                mtWeightInit,rawFeatureInputSize)
        print(net)

        local inputLookupTbl=  torch.Tensor(2,7):apply(
                function ()
                        return torch.random(1,10)
                end
        )
        inputLookupTbl[1][2] = 0
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


function testUseMaskzero()

        require 'rnn'
        require 'optim'

        inSize = 20
        batchSize = 2
        hiddenSize = 10
        seqLengthMax = 11
        numTargetClasses=5
        numSeq = 6

        x, y1 = {}, {}

        for i = 1, numSeq do
                local seqLength = torch.random(1,seqLengthMax)
                local temp = torch.zeros(seqLengthMax, inSize)
                local targets ={}
                if seqLength == seqLengthMax then
                        targets = (torch.rand(seqLength)*numTargetClasses):ceil()
                else
                        targets = torch.cat(torch.zeros(seqLengthMax-seqLength),(torch.rand(seqLength)*numTargetClasses):ceil())
                end
                temp[{{seqLengthMax-seqLength+1,seqLengthMax}}] = torch.randn(seqLength,inSize)

                table.insert(x, temp)
                table.insert(y1, targets)
                --print (temp, targets)

        end


        model = nn.Sequencer(
                nn.Sequential()
                        :add(nn.MaskZero(nn.FastLSTM(inSize,hiddenSize),1))
                        :add(nn.MaskZero(nn.Linear(hiddenSize, numTargetClasses),1))
                        :add(nn.MaskZero(nn.LogSoftMax(),1))
        )

        --        model = nn.Sequencer(
        --        nn.MaskZero(
        --                (nn.Sequential()
        --                        :add(nn.FastLSTM(inSize,hiddenSize))
        --                        :add(nn.Linear(hiddenSize, numTargetClasses))
        --                        :add(nn.LogSoftMax())),
        --                1
        --                )
        --        )

        print(model)

        --criterion = nn.SequencerCriterion(nn.MaskZero(nn.ClassNLLCriterion(),1))
        criterion = nn.SequencerCriterion(nn.MaskZeroCriterion(nn.ClassNLLCriterion(),1))

        output = model:forward(x)
        print(output[1])

        err = criterion:forward(output, y1)
        print(err)
end

function testModel2()

        local countSentence = 2
        local countWordOneSentence = 7

        local hiddenSize =20
        local nCountLabel = 9
        local rawDataInputSize = 40
        local rawFeatureInputSize = 15

        local inputData, inputDataFirst, outputData

        g_isUseFeatureWord = true
        g_isUseMaskZeroPadding = false
        g_nCountLabel = nCountLabel
        mtWeightInit = nil

        -- layer 1 - parse input
        local inputNet = GetInputEmbeddedLayer2(rawDataInputSize,hiddenSize,
                mtWeightInit,g_isUseFeatureWord,rawFeatureInputSize)

        -- random input data
        local inputData=  torch.Tensor(countSentence, countWordOneSentence):apply(
                function ()
                        return torch.random(1,rawDataInputSize)
                end
        )
        if(g_isUseMaskZeroPadding) then
                inputData[1][ (#inputData[1])[1] ] = 0
                inputData[1][ (#inputData[1])[1]-1 ] = 0
        end

        if(g_isUseFeatureWord) then
                local feature = torch.Tensor(countSentence, countWordOneSentence, rawFeatureInputSize):apply(
                        function ()
                                return torch.random(100) % 2
                        end
                )
                inputData = {inputData, {feature}}
        end

        inputDataFirst = inputData
        print(inputData)

        local net  = inputNet
        print(net)

        local outputData = net:forward(inputData)

        print((net.modules)[1].output[1])
        print((net.modules)[1].output[2])

        local _, outNetIdx = outputData:topk(1, true)

        print(outputData)
        print(outNetIdx)


        print '==============================================================='


        -- layer 2 - brnn
        local seqlstm = nn.SeqLSTM(hiddenSize+rawFeatureInputSize, math.ceil(hiddenSize/2))
        seqlstm.maskzero = true
        seqlstm.batchfirst = true

        local seqbrnn = nn.SeqBRNN(hiddenSize+rawFeatureInputSize, math.ceil(hiddenSize/2), true)
        seqbrnn.forwardModule.maskzero = true
        seqbrnn.backwardModule.maskzero = true
        print (seqbrnn)


        --        if(g_isUseMaskZeroPadding == true) then
        --                brnn.maskzero = true
        --        end

        net = seqbrnn

        inputData = outputData
        print (inputData)
        local outputData = net:forward(inputData)
        print(outputData)

        print '==============================================================='

        -- layer 3 - Linear
        net = nn.Sequencer(nn.MaskZero(nn.Linear( math.ceil(hiddenSize/2), g_nCountLabel), 1))

        inputData = outputData
        print (inputData)
        local outputData = net:forward(inputData)

        print '==============================================================='

        -- layer 4 - logsofmax
        net =  nn.Sequencer(nn.MaskZero(nn.LogSoftMax(),1))
        inputData = outputData
        print (inputData)
        local outputData = net:forward(inputData)
        print (outputData)

        print '==============================================================='
        net = InitModelNN("brnn",rawDataInputSize,hiddenSize,g_nCountLabel,
                mtWeightInit,rawFeatureInputSize)

        outNet = (net:forward(inputDataFirst))
        local _, outNetIdx = outNet:topk(1, true)
        print(outNetIdx)
end

function testModel3()

        local countSentence = 2
        local countWordOneSentence = 7

        local hiddenSize =20
        local nCountLabel = 9
        local rawDataInputSize = 40
        local rawFeatureInputSize = 15

        local inputData, inputDataFirst, outputData
        local inputFeature

        g_isUseFeatureWord = true
        g_isUseMaskZeroPadding = false
        g_nCountLabel = nCountLabel
        g_iModelTest = 2
        mtWeightInit = nil
        

        -- layer 1 - parse input
        local inputNet = GetInputEmbeddedLayer2(rawDataInputSize,hiddenSize,
                mtWeightInit,false,rawFeatureInputSize)
        local seqbrnn = nn.SeqBRNN(hiddenSize, math.ceil(hiddenSize/2), true)

        -- random input data
        local inputData=  torch.Tensor(countSentence, countWordOneSentence):apply(
                function ()
                        return torch.random(1,rawDataInputSize)
                end
        )
        if(g_isUseMaskZeroPadding) then
                inputData[1][ (#inputData[1])[1] ] = 0
                inputData[1][ (#inputData[1])[1]-1 ] = 0
        end

        if(g_isUseFeatureWord) then
                inputFeature = torch.Tensor(countSentence, countWordOneSentence, rawFeatureInputSize):apply(
                        function ()
                                return torch.random(100) % 2
                        end
                )
        end

        inputDataFirst = inputData
        print(inputData)

        local net  = inputNet
        print(net)

        local outputData = net:forward(inputData)

        print(outputData)


        print '==============================================================='


        -- layer 2 - brnn
        local brnnCover = nn.Sequential():add(inputNet):add(seqbrnn)
        local feature = nn.Sequencer(nn.Linear(rawFeatureInputSize, hiddenSize/2))
        local inputCover = nn.Sequential() 
                :add(nn.ParallelTable(): add(brnnCover) :add(feature))
                :add(nn.CAddTable())
        
        net = inputCover
        print(net)
        
        local outputData = net:forward({inputDataFirst, inputFeature})
        print(outputData)
        --assert(false, 'hi')
        print '==============================================================='

        -- layer 3 - Linear
        net = nn.Sequencer(--[[nn.MaskZero]](nn.Linear( math.ceil(hiddenSize/2), g_nCountLabel)--[[, 1]]))

        inputData = outputData
        print (inputData)
        local outputData = net:forward(inputData)

        print '==============================================================='

        -- layer 4 - logsofmax
        net =  nn.Sequencer(--[[nn.MaskZero]](nn.LogSoftMax()--[[,1]]))
        inputData = outputData
        print (inputData)
        local outputData = net:forward(inputData)
        print (outputData)

        print '==============================================================='
        
        g_iModelTest = 2
        g_isUseFeatureWord = true
        net = InitModelNN("brnnLstm",rawDataInputSize,hiddenSize,g_nCountLabel,
                mtWeightInit,rawFeatureInputSize)

        outNet = (net:forward({inputDataFirst, inputFeature}))
        print(outNet)
        local _, outNetIdx = outNet:topk(1, true)
        print(outNetIdx)
end

--testModel2()
--testUseMaskzero()
--testModel3()

