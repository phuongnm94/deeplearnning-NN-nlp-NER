require 'nn'

DataInputsCNNParser = {}

-- Derived class method new
function DataInputsCNNParser:new (o)
        o = o or {}
        setmetatable(o, self)
        self.__index = self

        return o
end

-- show hihi
function DataInputsCNNParser:show ()
        return 'hjihihi'
--        DataInputsCNNParser:new(o)
end

-- parse du lieu cau 
-- @function [parent=#DataInputsCNNParser] parseToWindowWordInput
function DataInputsCNNParser:parseToWindowWordInput (inputsSentence, targetsSentence, windowSize, paddingId)

        -- Check validate parameter
        assert(inputsSentence ~= nil)
        assert(targetsSentence ~= nil)
        assert(windowSize > 0 and windowSize%2 == 1)
        assert(paddingId ~= nil)


        local inputs, targets = {}, {}
        local nMaxCountPadding = math.floor(windowSize/2)
        local nCountInputsSentence = #inputsSentence

        -- Duyet qua tung cau
        for idxInputsSentence =1, nCountInputsSentence do

                -- Duyet qua tung word trong cau
                local idxWordInSentence, nCountWordInSentence = 1, #(inputsSentence[idxInputsSentence])

                for idxWordInSentence = 1, nCountWordInSentence do
                        if idxWordInSentence-nMaxCountPadding >= 1 and idxWordInSentence+nMaxCountPadding <= nCountWordInSentence then
                                inputs[#inputs+1] =
                                        torch.Tensor({table.unpack(inputsSentence[idxInputsSentence],
                                                idxWordInSentence-nMaxCountPadding, idxWordInSentence+nMaxCountPadding)})

                        else
                                local idxInVectorWord = 1
                                local idxInputs = #inputs+1
                                inputs[idxInputs] = torch.Tensor(windowSize):fill(paddingId)

                                for i = idxWordInSentence-nMaxCountPadding, idxWordInSentence+nMaxCountPadding do

                                        -- Neu i khong nam trong tap gia tri id cua cau => gia tri = padding
                                        if(i<1 or i>nCountWordInSentence) then goto _CONTINUE_NEXT_WORD end

                                        -- Neu vi tri i nam trong tap gia tri id cua cau 
                                        inputs[idxInputs][idxInVectorWord] = inputsSentence[idxInputsSentence][i]

                                        ::_CONTINUE_NEXT_WORD::
                                        idxInVectorWord = idxInVectorWord + 1
                                end
                        end
                        
                        targets[#targets + 1] = targetsSentence[idxInputsSentence][idxWordInSentence]
                end

        end

        ::_EXIT_FUNCTION::
        return inputs, targets
end

function tableUnpackExtend(tblSrc, posStart, posEnd)
        
        assert(tblSrc ~= nil)
        posStart = math.max(1,posStart)
        posEnd = math.min(#tblSrc,posEnd)
        
        local posCurrent  = posStart
        local size = 6000
        local tblRet = {}
        
        for posCurrent= posStart, posEnd, size do 
                tblRet[#tblRet + 1] = {table.unpack(tblSrc,posCurrent,math.min(posCurrent + size -1, posEnd))}
        end
        
        tblRet = nn.FlattenTable():forward(tblRet)
        return tblRet
end

function DataInputsCNNParser:genDataTrainTest (inputs, targets, fRateTrain, iDataset)
        
        -- Check validate parameter
        assert(inputs ~= nil)
        assert(targets ~= nil)
        assert(#targets == #inputs)
        assert(fRateTrain >= 0 and fRateTrain <= 1)
        assert(iDataset >= 1 and iDataset <= 10)
       
        local nCountInputs = #inputs 
        
        local nCountElementTest = math.max(1,math.floor(nCountInputs*(1-fRateTrain)))
        
        local nIndexStartTest = math.max(1,math.floor((nCountInputs*(1-fRateTrain)*(iDataset-1))) % nCountInputs)
        local nIndexEndTest = (math.min(nIndexStartTest+nCountElementTest, nCountInputs))%nCountInputs + 1
        
        local dataInputsTrain, dataInputsTest = {}, {} 
        local dataTargetsTrain, dataTargetsTest = {}, {} 
        local infoIndex = {nIndexStartTest, nIndexEndTest}
        
--        dataInputsTest = {table.unpack(inputs,nIndexStartTest,nIndexEndTest)}
--        dataInputsTrain = {{table.unpack(inputs,1,nIndexStartTest-1)}, {table.unpack(inputs,nIndexEndTest+1,nCountInputs)}}
--        dataInputsTrain = nn.FlattenTable():forward(dataInputsTrain)
--        
--        dataTargetsTest = {table.unpack(targets,nIndexStartTest,nIndexEndTest)}
--        dataTargetsTrain =  {{table.unpack(targets,1,nIndexStartTest-1)}, {table.unpack(targets,nIndexEndTest+1,nCountInputs)}}
--        dataTargetsTrain = nn.FlattenTable():forward(dataTargetsTrain)
        
        dataInputsTest = tableUnpackExtend(inputs,nIndexStartTest,nIndexEndTest)
        dataInputsTrain = {tableUnpackExtend(inputs,1,nIndexStartTest-1), tableUnpackExtend(inputs,nIndexEndTest+1,nCountInputs)}
        dataInputsTrain = nn.FlattenTable():forward(dataInputsTrain)
        
        dataTargetsTest = tableUnpackExtend(targets,nIndexStartTest,nIndexEndTest)
        dataTargetsTrain =  {tableUnpackExtend(targets,1,nIndexStartTest-1), tableUnpackExtend(targets,nIndexEndTest+1,nCountInputs)}
        dataTargetsTrain = nn.FlattenTable():forward(dataTargetsTrain)
        
        return dataInputsTest, dataInputsTrain, dataTargetsTest, dataTargetsTrain, infoIndex
end
