require('Testting')
require('InitData')


--[[
---
-- Training dataset
--
-- @function [parent=#global] TrainningEachBatchSentence(rnn, criterion, inputs, targets)
-- @param rnn mang can hoc
-- @param criterion ham Loss
-- @param inputs Tap du lieu dau vao
-- @param targets Tap du lieu muc tieu
--
function TrainningEachBatchSentence(rnn, criterion, inputs, targets)

        -- ---------------------------------------------------------------------------------------
        -- ---------------------------------------------------------------------------------------
        -- TRAINING
        -- ---------------------------------------------------------------------------------------
        -- ---------------------------------------------------------------------------------------
        local iteration = 1
        local idxBatch, sizeBatch = 1, 20

        local inputTest, targetTest = {}, {}
        local countLoopForOneBatch = 5

        thresholdTraining =  math.ceil((#inputs*0.8) + 1)
        local mtRateClassTraining = (GetRateTrainingEachClass(targets,1,thresholdTraining))
        criterion.criterion.weights = mtRateClassTraining

        for k = 1, 20 do

                print(' Loop all data i = ' .. k .. '/20')


                iteration = 1
                while iteration < thresholdTraining do

                        -- ----------------------------------------------------------------
                        -- Khoi tao du lieu input cho 1 cau
                        -- ----------------------------------------------------------------

                        local sentence = {}
                        local sentenceNERDist = {}
                        local batchInput, batchTarget = {}, {}

                        local outputs, err
                        local gradOutputs, gradInputs


                        -- --------------------------------
                        -- tao 20 cau trong bo batch inputs
                        idxBatch = 1
                        while idxBatch <= sizeBatch do

                                sentence, sentenceNERDist = {} , {}

                                local wordCountInSentence = #inputs[iteration]

                                for i = 1, wordCountInSentence do
                                        if(inputs[iteration][i] == nil) then
                                                print('[training] Err 1 - iter' .. iteration)
                                                goto CONTINUE
                                        end
                                        if(targets[iteration][i] == nil) then
                                                print('[training] Err 2 - iter' .. iteration)
                                                --                                                targets[iteration][i] = 1
                                                goto CONTINUE
                                        end


                                        sentence[i] = torch.LongTensor(1):fill(inputs[iteration][i])

                                        sentenceNERDist[i] = torch.LongTensor(1):fill(targets[iteration][i])

                                end

                                batchInput [idxBatch] = sentence
                                batchTarget[idxBatch] = sentenceNERDist
                                idxBatch = idxBatch + 1

                                ::CONTINUE::
                                iteration = iteration + 1
                                if(iteration > thresholdTraining) then
                                        break
                                end


                        end


                        for j = 1, countLoopForOneBatch do

                                for key, sentence in pairs(batchInput) do


                                        sentenceNERDist = batchTarget[key]

                                        -- ----------------------------------------------------------------
                                        -- 2. forward sequence through rnn
                                        -- ----------------------------------------------------------------
                                        rnn:zeroGradParameters()

                                        --                print(rnn:get(1):get(1):get(1):get(2).weight[idWordTest])
                                        --                print('check weight after')

                                        outputs = rnn:forward(sentence)
                                        err = criterion:forward(outputs, sentenceNERDist)


                                        -- ----------------------------------------------------------------
                                        -- 3. backward sequence through rnn (i.e. backprop through time)
                                        -- ----------------------------------------------------------------
                                        gradOutputs = criterion:backward(outputs, sentenceNERDist)
                                        gradInputs = rnn:backward(sentence, gradOutputs)


                                        -- ----------------------------------------------------------------
                                        -- 4. update weights
                                        -- ----------------------------------------------------------------
                                        rnn:updateParameters(lr)

                                        idxBatch = idxBatch + 1
                                end


                        end

                        print(string.format("[%d] Iteration %d ; NLL err = %f ", k, iteration, err))


                end

                print(' Testting ' .. k .. '/20')
                testing(rnn,inputs,targets,g_nCountLabel)
        end
end
]]

--[[
---
-- Training dataset
--
-- @function [parent=#global] TrainningEachBatchSentence(rnn, criterion, inputs, targets)
-- @param rnn mang can hoc
-- @param criterion ham Loss
-- @param inputs Tap du lieu dau vao
-- @param targets Tap du lieu muc tieu
-- @param nRate = ti le tap hoc
--
function TrainningUseCrossvalidation(rnn, criterion, inputs, targets, nRate)

        -- ---------------------------------------------------------------------------------------
        -- ---------------------------------------------------------------------------------------
        -- TRAINING
        -- ---------------------------------------------------------------------------------------
        -- ---------------------------------------------------------------------------------------
        local iteration = 1
        local idxBatch, sizeBatch = 1, 20

        local inputTest, targetTest = {}, {}
        local countLoopForOneBatch = 5

        local nSizeInput = #inputs
        local nIndexStart = 1
        local nIndexEnd = nSizeInput

        -- validate input nRate
        if nRate == nil or nRate <= 0 or nRate >= 1 then nRate = 0.9 end

        thresholdTraining =  math.ceil((#inputs*nRate) + 1)
        g_result = {}
        --        thresholdTraining = 100


        local k = g_iDataset

        -- Tinh toan lai gioi han chi so cau tap du lieu hoc
        nIndexStart = math.ceil(((k-1)*(1-nRate)*nSizeInput))%nSizeInput + 1
        nIndexEnd = (thresholdTraining + nIndexStart )%nSizeInput + 1
        print(string.format('Dataset [%d/%d] : Cau %d - Cau %d', k, 10, nIndexStart, nIndexEnd))

        -- Tinh ma tran trong so khi tap hoc thay doi
        local mtRateClassTraining = (GetRateTrainingEachClass(targets,nIndexStart,thresholdTraining))
        criterion.criterion.weights = mtRateClassTraining


        -- lap tren tung bo du lieu nhieu lan
        local idxLoopOneDataset, nCountLoopOneDataset = 1, 30

        for idxLoopOneDataset = 1, nCountLoopOneDataset do


                iteration = nIndexStart
                while iteration ~= nIndexEnd do

                        -- ----------------------------------------------------------------
                        -- Khoi tao du lieu input cho 1 cau
                        -- ----------------------------------------------------------------

                        local sentence = {}
                        local sentenceNERDist = {}
                        local batchInput, batchTarget = {}, {}

                        local outputs, err
                        local gradOutputs, gradInputs


                        -- --------------------------------
                        -- tao 20 cau trong bo batch inputs
                        idxBatch = 1
                        while idxBatch <= sizeBatch do

                                sentence = torch.LongTensor({inputs[iteration]})

                                sentenceNERDist = torch.LongTensor({targets[iteration]})

                                batchInput [idxBatch] = sentence
                                batchTarget[idxBatch] = sentenceNERDist
                                idxBatch = idxBatch + 1

                                ::CONTINUE::
                                iteration = iteration%nSizeInput + 1
                                if(iteration == nIndexEnd) then
                                        break
                                end


                        end


                        for j = 1, countLoopForOneBatch do

                                for key, sentence in pairs(batchInput) do


                                        sentenceNERDist = batchTarget[key]

                                        -- ----------------------------------------------------------------
                                        -- 2. forward sequence through rnn
                                        -- ----------------------------------------------------------------
                                        rnn:zeroGradParameters()


                                        outputs = rnn:forward(sentence)
                                        err = criterion:forward(outputs, sentenceNERDist)


                                        -- ----------------------------------------------------------------
                                        -- 3. backward sequence through rnn (i.e. backprop through time)
                                        -- ----------------------------------------------------------------
                                        gradOutputs = criterion:backward(outputs, sentenceNERDist)
                                        gradInputs = rnn:backward(sentence, gradOutputs)


                                        -- ----------------------------------------------------------------
                                        -- 4. update weights
                                        -- ----------------------------------------------------------------
                                        rnn:updateParameters(lr)

                                        idxBatch = idxBatch + 1
                                end

                        end

                        if(iteration%100 == 1) then
                                print(string.format("[Data - %d - loop: %d] Cau %d ; NLL err = %f ", k, idxLoopOneDataset, iteration, err))
                        end


                end

                print(string.format("[Data - %d] Testing %d / %d ", k, idxLoopOneDataset, nCountLoopOneDataset))
                g_result[k] = TestUseCrossvalidation(rnn,inputs,targets,g_nCountLabel, nIndexStart, nIndexEnd)
        end


        return nIndexStart, nIndexEnd
end
]]

--[[
---
-- Training dataset
--
-- @function [parent=#global] TrainningUseOptimBatchCrossvalidation(rnn, criterion, inputs, targets)
-- @param rnn mang can hoc
-- @param criterion ham Loss
-- @param inputs Tap du lieu dau vao
-- @param targets Tap du lieu muc tieu
-- @param nRate = ti le tap hoc
--
function TrainningUseOptimBatchFeaturesCrossvalidation2(rnn, criterion, inputs, targets,features, nRate)

        -- ---------------------------------------------------------------------------------------
        -- ---------------------------------------------------------------------------------------
        -- TRAINING
        -- ---------------------------------------------------------------------------------------
        -- ---------------------------------------------------------------------------------------
        local iteration = 1
        local idxBatch, sizeBatch = 1, 20

        local inputTest, targetTest = {}, {}
        local countLoopForOneBatch = nil
        local nCountLoopOneDataset = nil

        local nSizeInput = #inputs
        local nIndexStart = 1
        local nIndexEnd = nSizeInput

        -- validate input option
        if nRate == nil or nRate <= 0 or nRate >= 1 then nRate = 0.9 end
        if g_countLoopForOneBatch == nil then countLoopForOneBatch =5 else countLoopForOneBatch = g_countLoopForOneBatch end
        if g_countLoopAllData == nil then nCountLoopOneDataset = 30 else nCountLoopOneDataset = g_countLoopAllData end


        thresholdTraining =  #inputs -- math.ceil((#inputs*nRate) + 1)

        local k = g_iDataset

        nIndexStart = 1 --math.ceil(((k-1)*(1-nRate)*nSizeInput))%nSizeInput + 1
        nIndexEnd = thresholdTraining --(thresholdTraining + nIndexStart )%nSizeInput
        --nIndexEnd = nIndexStart + 10
        print(string.format('Dataset [%d/%d] : Cau %d - Cau %d', k, 10, nIndexStart, nIndexEnd))


        -- Tinh ma tran trong so khi tap hoc thay doi
        local mtRateClassTraining = (GetRateTrainingEachClass(targets,nIndexStart,thresholdTraining))
        criterion.criterion.weights = mtRateClassTraining


        -- lap tren tung bo du lieu nhieu lan
        for idxLoopOneDataset = 1, nCountLoopOneDataset do

                iteration = nIndexStart
                for iteration =nIndexStart, nIndexEnd do -- while iteration ~= nIndexEnd do

                        -- ----------------------------------------------------------------
                        -- Khoi tao du lieu input cho 1 cau
                        -- ----------------------------------------------------------------

                        local sentence, sentenceNERDist, sentenceFeatures = {}, {}, {}
                        local batchInput, batchTarget = {}, {}
                        local nCountSentence = nil

                        local outputs, err
                        local gradOutputs, gradInputs



                        sentence = torch.Tensor(inputs[iteration]):t()

                        sentenceNERDist = torch.Tensor(targets[iteration]):t()
                        
                        sentenceFeatures = torch.Tensor(features[iteration])
                         
                        data["inputs"], data["targets"] = {sentence, sentenceFeatures}, sentenceNERDist
                        
                        nCountSentence = sentence:size()[1]

                        iteration = iteration%nSizeInput + 1



                        for j = 1, countLoopForOneBatch do

                                -- train a mini_batch of batchSize in parallel
                                _, fs = optim.sgd(feval,x, sgd_params)

                        end
                        
                        if(iteration%100 == 0) then
                                print(string.format("[Data - %d - loop: %d/%d] Cau %d ; NLL err = %f ", 
                                        k, idxLoopOneDataset, nCountLoopOneDataset, iteration, fs[1] / nCountSentence))
                        end


                end

                print(string.format('Dataset [%d/10] Testing in loop - %d / %d', k , idxLoopOneDataset, nCountLoopOneDataset))
                g_result[k] = TestUseCrossvalidationParallel(rnn,inputs,targets,g_nCountLabel, nIndexStart, nIndexEnd,features)
        end

        ::EXIT_FUNCTION::
        return nIndexStart, nIndexEnd

end
]]

---
-- Training dataset
--
-- @function [parent=#global] TrainningEachBatchSentence(rnn, criterion, inputs, targets)
-- @param rnn mang can hoc
-- @param criterion ham Loss
-- @param inputs Tap du lieu dau vao
-- @param targets Tap du lieu muc tieu
-- @param nRate = ti le tap hoc
--
function TrainningUseCrossvalidationParallel(rnn, criterion, inputs, targets, nRate)

        -- ---------------------------------------------------------------------------------------
        -- ---------------------------------------------------------------------------------------
        -- TRAINING
        -- ---------------------------------------------------------------------------------------
        -- ---------------------------------------------------------------------------------------
        local iteration = 1
        local idxBatch, sizeBatch = 1, 20

        local inputTest, targetTest = {}, {}
        local countLoopForOneBatch = nil
        local nCountLoopOneDataset = nil


        local nSizeInput = #inputs
        local nIndexStart = 1
        local nIndexEnd = nSizeInput

        -- validate input option
        if nRate == nil or nRate <= 0 or nRate >= 1 then nRate = 0.9 end
        if g_countLoopForOneBatch == nil then countLoopForOneBatch =5 else countLoopForOneBatch = g_countLoopForOneBatch end
        if g_countLoopAllData == nil then nCountLoopOneDataset = 30 else nCountLoopOneDataset = g_countLoopAllData end


        thresholdTraining =  #inputs -- math.ceil((#inputs*nRate) + 1)

        local k = g_iDataset

        nIndexStart = 1 --math.ceil(((k-1)*(1-nRate)*nSizeInput))%nSizeInput + 1
        nIndexEnd = thresholdTraining --(thresholdTraining + nIndexStart )%nSizeInput
        print(string.format('Dataset [%d/%d]', k, 10))


        -- Tinh ma tran trong so khi tap hoc thay doi
        --        local mtRateClassTraining = (GetRateTrainingEachClass(targets,nIndexStart,thresholdTraining))
        --        criterion.criterion.weights = mtRateClassTraining
        for idxLoopOneDataset = 1, nCountLoopOneDataset do
                
                local indicates = torch.randperm(nIndexEnd)
                
                for idx = nIndexStart, nIndexEnd do

                        iteration = indicates[idx]
                        
                        -- ----------------------------------------------------------------
                        -- Khoi tao du lieu input cho 1 cau
                        -- ----------------------------------------------------------------

                        local sentence = {}
                        local sentenceNERDist = {}
                        local batchInput, batchTarget = {}, {}
                        local nCountSentence = nil

                        local outputs, err
                        local gradOutputs, gradInputs


                        sentence = torch.Tensor(inputs[iteration])

                        sentenceNERDist = torch.Tensor(targets[iteration])

                        nCountSentence = sentence:size()[1]

                        iteration = iteration%nSizeInput + 1



                        for j = 1, countLoopForOneBatch do

                                -- ----------------------------------------------------------------
                                -- 2. forward sequence through rnn
                                -- ----------------------------------------------------------------
                                rnn:zeroGradParameters()

                                outputs = rnn:forward(sentence)
                                

                                err = criterion:forward(outputs, sentenceNERDist)
                                print(outputs)
                                print(sentenceNERDist)
                                assert(false, 'end to debug')

                                -- ----------------------------------------------------------------
                                -- 3. backward sequence through rnn (i.e. backprop through time)
                                -- ----------------------------------------------------------------
                                gradOutputs = criterion:backward(outputs, sentenceNERDist)
                                gradInputs = rnn:backward(sentence, gradOutputs)


                                -- ----------------------------------------------------------------
                                -- 4. update weights
                                -- ----------------------------------------------------------------
                                rnn:updateParameters(lr)

                        end


                        if(iteration%100 == 1) then
                                print(string.format("[Data - %d - loop: %d/%d] Cau %d ; NLL err = %f ", 
                                        k, idxLoopOneDataset,nCountLoopOneDataset, iteration, err))
                        end


                end

                print(string.format('Dataset [%d/10] Testing in loop - %d/%d', k , idxLoopOneDataset,nCountLoopOneDataset))
                g_result[k] = TestUseCrossvalidationParallel(rnn,inputs,targets,g_nCountLabel, nIndexStart, nIndexEnd)
        end

        ::EXIT_FUNCTION::
        return nIndexStart, nIndexEnd

end


function InitOptimizeConfig(netBRNN, opt)

        -- for optimize

        -- get weights and loss wrt weights from the model
        x, dl_dx = netBRNN:getParameters()

        -- In the following code, we define a closure, feval, which computes
        -- the value of the loss function at a given point x, and the gradient of
        -- that function with respect to x. weigths is the vector of trainable weights,
        -- it extracts a mini_batch via the nextBatch method

        sgd_params = {
                learningRate = opt.lr,
                learningRateDecay = 1e-4,
                weightDecay = 0,
                momentum = opt.momentum
        }

        sgd_params1 = {
                learningRate = 1e-2,
                learningRateDecay = 1e-4,
                weightDecay = 1e-3,
                momentum = 1e-4
        }

        data = {
                inputs = nil,
                targets = nil
        }

        feval = function(x_new)
                        
                        -- copy the weight if are changed
                        if x ~= x_new then
                                x:copy(x_new)
                        end
        
                        -- select a training batch
                        local inputs, targets = data["inputs"], data["targets"]
        
                        -- reset gradients (gradients are always accumulated, to accommodate
                        -- batch methods)
                        dl_dx:zero()
        
                        -- evaluate the loss function and its derivative wrt x, given mini batch
                        local prediction = netBRNN:forward(inputs)
                        local loss_x = criterion:forward(prediction, targets)
                        netBRNN:backward(inputs, criterion:backward(prediction, targets))
        
                        return loss_x, dl_dx
                end
end



---
-- Training dataset
--
-- @function [parent=#global] TrainningUseOptimBatchCrossvalidation(rnn, criterion, inputs, targets)
-- @param rnn mang can hoc
-- @param criterion ham Loss
-- @param inputs Tap du lieu dau vao
-- @param targets Tap du lieu muc tieu
-- @param nRate = ti le tap hoc
--
function TrainningUseOptimBatchCrossvalidation(rnn, criterion, inputs, targets, nRate)

        -- ---------------------------------------------------------------------------------------
        -- ---------------------------------------------------------------------------------------
        -- TRAINING
        -- ---------------------------------------------------------------------------------------
        -- ---------------------------------------------------------------------------------------
        local iteration = 1
        local idxBatch, sizeBatch = 1, 20

        local inputTest, targetTest = {}, {}
        local countLoopForOneBatch = nil
        local nCountLoopOneDataset = nil

        local nSizeInput = #inputs
        local nIndexStart = 1
        local nIndexEnd = nSizeInput

        -- validate input option
        if nRate == nil or nRate <= 0 or nRate >= 1 then nRate = 0.9 end
        if g_countLoopForOneBatch == nil then countLoopForOneBatch =5 else countLoopForOneBatch = g_countLoopForOneBatch end
        if g_countLoopAllData == nil then nCountLoopOneDataset = 30 else nCountLoopOneDataset = g_countLoopAllData end


        thresholdTraining =  #inputs -- math.ceil((#inputs*nRate) + 1)

        local k = g_iDataset

        nIndexStart = 1 --math.ceil(((k-1)*(1-nRate)*nSizeInput))%nSizeInput + 1
        nIndexEnd = thresholdTraining --(thresholdTraining + nIndexStart )%nSizeInput
        print(string.format('Dataset [%d/%d] : ', k, 10))


        -- Tinh ma tran trong so khi tap hoc thay doi
        --local mtRateClassTraining = (GetRateTrainingEachClass(targets,nIndexStart,thresholdTraining))
        --criterion.criterion.weights = mtRateClassTraining


        -- lap tren tung bo du lieu nhieu lan
        for idxLoopOneDataset = 1, nCountLoopOneDataset do
                
                local indicates = torch.randperm(nIndexEnd)
                
                for idx = nIndexStart, nIndexEnd do --while iteration ~= nIndexEnd do
                        
                        iteration = indicates[idx]
                        
                        -- ----------------------------------------------------------------
                        -- Khoi tao du lieu input cho 1 cau
                        -- ----------------------------------------------------------------

                        local sentence, sentenceNERDist, sentenceFeatures = {}, {}, {}
                        local batchInput, batchTarget = {}, {}
                        local nCountSentence = nil

                        local outputs, err
                        local gradOutputs, gradInputs

                        sentence = torch.Tensor(inputs[iteration])

                        sentenceNERDist = torch.Tensor(targets[iteration])
                                                

                        data["inputs"], data["targets"] = sentence, sentenceNERDist

                        nCountSentence = sentence:size()[1]

                        iteration = iteration%nSizeInput + 1

                        
                        for j = 1, countLoopForOneBatch do

                                -- train a mini_batch of batchSize in parallel
                                _, fs = optim.sgd(feval,x, sgd_params)

                        end


                        if(iteration%100 == 0) then
                                collectgarbage()
                                print(string.format("[Data - %d - loop: %d/%d] Cau %d ; NLL err = %f ", 
                                        k, idxLoopOneDataset, nCountLoopOneDataset, iteration, fs[1] / nCountSentence))
                        end
                        
                        data["inputs"], data["targets"] = nil, nil 

                end

                print(string.format('Dataset [%d/10] Testing in loop - %d / %d', k , idxLoopOneDataset, nCountLoopOneDataset))
                g_result[k] = TestUseCrossvalidationParallel(rnn,DataSetGroup["inputsTest"],DataSetGroup["targetsTest"],g_nCountLabel, nIndexStart, nIndexEnd)
        end

        ::EXIT_FUNCTION::
        return nIndexStart, nIndexEnd

end

--[[
---
-- Training dataset
--
-- @function [parent=#global] TrainningUseOptimBatchCrossvalidation(rnn, criterion, inputs, targets)
-- @param rnn mang can hoc
-- @param criterion ham Loss
-- @param inputs Tap du lieu dau vao
-- @param targets Tap du lieu muc tieu
-- @param nRate = ti le tap hoc
--


function TrainningUseOptimBatchCrossvalidation2(rnn, criterion, inputs, targets, nRate)

        -- ---------------------------------------------------------------------------------------
        -- ---------------------------------------------------------------------------------------
        -- TRAINING
        -- ---------------------------------------------------------------------------------------
        -- ---------------------------------------------------------------------------------------
        local iteration = 1
        local idxBatch, sizeBatch = 1, 20

        local inputTest, targetTest = {}, {}
        local countLoopForOneBatch = nil
        local nCountLoopOneDataset = nil

        local nSizeInput = #inputs
        local nIndexStart = 1
        local nIndexEnd = nSizeInput

        -- validate input option
        if nRate == nil or nRate <= 0 or nRate >= 1 then nRate = 0.9 end
        if g_countLoopForOneBatch == nil then countLoopForOneBatch =5 else countLoopForOneBatch = g_countLoopForOneBatch end
        if g_countLoopAllData == nil then nCountLoopOneDataset = 30 else nCountLoopOneDataset = g_countLoopAllData end


        thresholdTraining =  math.ceil((#inputs*nRate) + 1)

        local k = g_iDataset

        nIndexStart = math.ceil(((k-1)*(1-nRate)*nSizeInput))%nSizeInput + 1
        nIndexEnd = (thresholdTraining + nIndexStart )%nSizeInput
        print(string.format('Dataset [%d/%d] : Cau %d - Cau %d', k, 10, nIndexStart, nIndexEnd))


        -- Tinh ma tran trong so khi tap hoc thay doi
        local mtRateClassTraining = (GetRateTrainingEachClass(targets,nIndexStart,thresholdTraining))
        criterion.criterion.weights = mtRateClassTraining


        -- lap tren tung bo du lieu nhieu lan
        for idxLoopOneDataset = 1, nCountLoopOneDataset do

                iteration = nIndexStart
                while iteration ~= nIndexEnd do

                        -- ----------------------------------------------------------------
                        -- Khoi tao du lieu input cho 1 cau
                        -- ----------------------------------------------------------------

                        local sentence, sentenceNERDist, sentenceFeatures = {}, {}, {}
                        local batchInput, batchTarget = {}, {}
                        local nCountSentence = nil

                        local outputs, err
                        local gradOutputs, gradInputs



                        sentence = torch.Tensor(inputs[iteration]):t()

                        sentenceNERDist = torch.Tensor(targets[iteration]):t()
                                                

                        data["inputs"], data["targets"] = sentence, sentenceNERDist

                        nCountSentence = sentence:size()[1]

                        iteration = iteration%nSizeInput + 1

                        
                        for j = 1, countLoopForOneBatch do

                                -- train a mini_batch of batchSize in parallel
                                _, fs = optim.sgd(feval,x, sgd_params)

                        end


                        if(iteration%100 == 0) then
                                print(string.format("[Data - %d - loop: %d/%d] Cau %d ; NLL err = %f ", 
                                        k, idxLoopOneDataset, nCountLoopOneDataset, iteration, fs[1] / nCountSentence))
                        end
                        
                        data["inputs"], data["targets"] = nil, nil 

                end

                print(string.format('Dataset [%d/10] Testing in loop - %d / %d', k , idxLoopOneDataset, nCountLoopOneDataset))
                g_result[k] = TestUseCrossvalidationParallel(rnn,inputs,targets,g_nCountLabel, nIndexStart, nIndexEnd)
        end

        ::EXIT_FUNCTION::
        return nIndexStart, nIndexEnd

end
]]

---
-- Training dataset
--
-- @function [parent=#global] TrainningUseOptimBatchCrossvalidation(rnn, criterion, inputs, targets)
-- @param rnn mang can hoc
-- @param criterion ham Loss
-- @param inputs Tap du lieu dau vao
-- @param targets Tap du lieu muc tieu
-- @param nRate = ti le tap hoc
--
function TrainningUseOptimBatchFeaturesCrossvalidation(rnn, criterion, inputs, targets,features, nRate)

        -- ---------------------------------------------------------------------------------------
        -- ---------------------------------------------------------------------------------------
        -- TRAINING
        -- ---------------------------------------------------------------------------------------
        -- ---------------------------------------------------------------------------------------
        local iteration = 1
        local idxBatch, sizeBatch = 1, 20

        local inputTest, targetTest = {}, {}
        local countLoopForOneBatch = nil
        local nCountLoopOneDataset = nil

        local nSizeInput = #inputs
        local nIndexStart = 1
        local nIndexEnd = nSizeInput

        -- validate input option
        if nRate == nil or nRate <= 0 or nRate >= 1 then nRate = 0.9 end
        if g_countLoopForOneBatch == nil then countLoopForOneBatch =5 else countLoopForOneBatch = g_countLoopForOneBatch end
        if g_countLoopAllData == nil then nCountLoopOneDataset = 30 else nCountLoopOneDataset = g_countLoopAllData end


        thresholdTraining =  #inputs 

        local k = g_iDataset

        nIndexStart = 1 
        nIndexEnd = thresholdTraining 
        print(string.format('Dataset [%d/%d] : ', k, 10))
       
       
        -- Tinh ma tran trong so khi tap hoc thay doi
        --local mtRateClassTraining = (GetRateTrainingEachClass(targets,nIndexStart,thresholdTraining))
        --criterion.criterion.weights = mtRateClassTraining


        -- lap tren tung bo du lieu nhieu lan
        for idxLoopOneDataset = 1, nCountLoopOneDataset do
                
                local indicates = torch.randperm(nIndexEnd)
                
                for idx = nIndexStart, nIndexEnd do

                        iteration = indicates[idx]
                        
                        -- ----------------------------------------------------------------
                        -- Khoi tao du lieu input cho 1 cau
                        -- ----------------------------------------------------------------

                        local sentence, sentenceNERDist, sentenceFeatures = {}, {}, {}
                        local batchInput, batchTarget = {}, {}
                        local nCountSentence = nil

                        local outputs, err
                        local gradOutputs, gradInputs



                        sentence = torch.Tensor(inputs[iteration])

                        sentenceNERDist = torch.Tensor(targets[iteration])
                        
                        sentenceFeatures = torch.Tensor(features[iteration])
                        
                        if(g_iModelTest == 1) then 
                                data["inputs"], data["targets"] = {sentence, {sentenceFeatures}}, sentenceNERDist
                        else
                                data["inputs"], data["targets"] = {sentence, sentenceFeatures}, sentenceNERDist
                        end
                        nCountSentence = sentence:size()[1]

                        for j = 1, countLoopForOneBatch do

                                -- train a mini_batch of batchSize in parallel
                                _, fs = optim.sgd(feval,x, sgd_params)

                        end
                        
                        if(iteration%100 == 0) then
                                collectgarbage()
                                print(string.format("[Data - %d - loop: %d/%d] Cau %d ; NLL err = %f ", 
                                        k, idxLoopOneDataset, nCountLoopOneDataset, iteration, fs[1] / nCountSentence))
                        end


                end

                print(string.format('Dataset [%d/10] Testing in loop - %d / %d', k , idxLoopOneDataset, nCountLoopOneDataset))
                g_result[k] = TestUseCrossvalidationParallel(
                        rnn,
                        DataSetGroup["inputsTest"],
                        DataSetGroup["targetsTest"],
                        g_nCountLabel, 
                        nIndexStart, nIndexEnd,
                        DataSetGroup["featuresTest"])
        end

        ::EXIT_FUNCTION::
        return nIndexStart, nIndexEnd

end

