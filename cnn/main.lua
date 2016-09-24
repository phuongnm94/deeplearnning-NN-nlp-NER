
require("nn")
require("nninit")
require 'optim'
require("csvigo")
require('DataInputsCNNParser')
require('ParamsParser')
require('InitData')

-- ---------------------------------------------------------------------------------------
-- ---------------------------------------------------------------------------------------
-- INIT DATA SET AND DICTIONARY
-- ---------------------------------------------------------------------------------------
-- ---------------------------------------------------------------------------------------

local opt = ParamsParser()

lr = opt.lr
iDataSet = opt.iDataset
fTrainRate = opt.trainRate

print(opt)


rawDataInputSize = 25000
g_wordPaddingInfo = {}
g_wordPaddingInfo["size"] = 1
g_wordPaddingInfo["lstid"]={}

InitData('SubDict_vc.txt','NonTag4type.tag')

windowSize, paddingId = 11, g_wordPaddingInfo["lstid"][1]

dataParser = DataInputsCNNParser:new(nil)
inputsModel, targetsModel = dataParser:parseToWindowWordInput(inputs, targets, windowSize, paddingId)


dataInputsTest, dataInputsTrain, dataTargetsTest, dataTargetsTrain, infoIndex
= dataParser:genDataTrainTest (inputsModel, targetsModel, fTrainRate, iDataSet)

-- chuyen hoa data
dictSize = rawDataInputSize
csv_tensor = mtWeightInit
embeddedSize = ((#csv_tensor)[2])

trainingSize = #dataInputsTrain
testingSize = #dataInputsTest

data = dataInputsTrain
labels = dataTargetsTrain

data_test = dataInputsTest
labels_test = dataTargetsTest

classes = {"O", "B-LOC", "B-PER", "B-ORG", "B-TOUR", "I-ORG", "I-PER", "I-TOUR", "I-LOC", "B-PRO", "I-PRO"}
nCountTopic = 9 -- #classes



goto _INIT_MODEL

-- load trainning_size
matrix_words = csvigo.load{path='pre-processing/train_data/full_train_matrix_file.txt', mode='large', separator=' '}
classes = {"O", "B-LOC", "B-PER", "B-ORG", "B-TOUR", "I-ORG", "I-PER", "I-TOUR", "I-LOC", "B-PRO", "I-PRO"}

windowSize = #matrix_words[1]
trainingSize = #matrix_words

data = torch.Tensor(trainingSize, windowSize)
for i=1, trainingSize do
        for j=1, windowSize do
                data[i][j] = tonumber(matrix_words[i][j]) + 1 -- lua index start from 1, not 0
        end
end


-- read labels
labelsRaw = csvigo.load{path='pre-processing/train_data/full_train_label_file.txt', mode='large', separator=' '}
labels = torch.DoubleTensor(#labelsRaw)
for i=1, #labelsRaw do
        labels[i] = tonumber(labelsRaw[i][1]) + 1
end

print(string.format("window_size = %d, training_size = %d", windowSize, trainingSize))



-- load test_data --
matrix_words_test = csvigo.load{path='pre-processing/train_data/full_test_matrix_file.txt', mode='large', separator=' '}
testingSize = #matrix_words_test

data_test = torch.Tensor(testingSize, windowSize)
for i=1, testingSize do
        for j=1, windowSize do
                data_test[i][j] = tonumber(matrix_words_test[i][j]) + 1 -- lua index start from 1, not 0
        end
end


-- read labels --
labelsRawTest = csvigo.load{path='pre-processing/train_data/full_test_label_file.txt', mode='large', separator=' '}
labels_test = torch.DoubleTensor(#labelsRawTest)
for i=1, #labelsRawTest do
        labels_test[i] = tonumber(labelsRawTest[i][1]) + 1
end

print(string.format("window_size = %d, testing_size = %d", windowSize, testingSize))

-- load embedded
w2v_mat = csvigo.load{path='pre-processing/train_data/total_matrix_final.txt', mode='large', separator=' '}

rows = #w2v_mat
cols = #w2v_mat[1] - 1 -- the last elem is \n

-- load word
words = csvigo.load{path='pre-processing/train_data/total_words_final.txt', mode='large', separator=' '}
dictSize = #words
embeddedSize = cols

print (string.format("Size {rows=%d, cols=%d}", rows, cols))
print (string.format("Number of total words = %d", dictSize))



-- create tensor --
csv_tensor = torch.Tensor(rows, cols)
print(string.format("Generate new tensor at size %d x %d", rows, cols))

for i=1, rows do
        for j=1, cols do
                csv_tensor[i][j] = tonumber(w2v_mat[i][j])
        end
end

::_INIT_MODEL::
lookup = nn.LookupTable(dictSize, embeddedSize)
lookup.weight = csv_tensor

print(#lookup:forward(data[1]))

-- some matrix layer --
L = 100
K = embeddedSize

windowSize = 11
model = nn.Sequential()
model:add(lookup)
model:add(nn.Reshape(K*windowSize))
model:add(nn.Linear(K*windowSize, L))
model:add(nn.HardTanh())
model:add(nn.Linear(L,nCountTopic))
model:add(nn.LogSoftMax())

print(model)


-- generate weight for criterion --
weight = torch.Tensor(nCountTopic)
for i=1,nCountTopic do
        weight[i] = 0
end

for i=1, #labels do
        weight[labels[i]] = weight[labels[i]] + 1
end

for i=1,nCountTopic do
        weight[i] = #labels / weight[i]
end

print(weight)



criterion = nn.ClassNLLCriterion(weight)
x, dl_dx = model:getParameters()

sgd_params = {
        learningRate = lr,
        learningRateDecay = 1e-4,
        weightDecay = 1e-3,
        momentum = 1e-4
}


print(trainingSize)
function step(batch_size)
        local current_loss = 0
        local count = 0
        local shuffle = torch.randperm(trainingSize)
        batch_size = batch_size or 200

        for t = 0, trainingSize, batch_size do
                if t>=trainingSize then
                        break
                end

                -- setup inputs and targets for this mini-batch
                local size = math.min(t + batch_size, trainingSize) - t
                local inputs = torch.Tensor(size, windowSize)
                local targets = torch.Tensor(size)
                for i = 1, size do
                        local input = data[shuffle[i+t]]
                        local target = labels[shuffle[i+t]]
                        -- if target == 0 then target = 10 end
                        inputs[i] = input
                        targets[i] = target
                end

                local feval = function(x_new)
                        -- reset data
                        if x ~= x_new then x:copy(x_new) end
                        dl_dx:zero()

                        -- perform mini-batch gradient descent
                        outputs = model:forward(inputs);
                        local loss = criterion:forward(model:forward(inputs), targets)
                        model:backward(inputs, criterion:backward(model.output, targets))

                        return loss, dl_dx
                end

                _, fs = optim.sgd(feval, x, sgd_params)
                -- fs is a table containing value of the loss function
                -- (just 1 value for the SGD optimization)
                count = count + 1
                current_loss = current_loss + fs[1]
        end

        -- normalize loss
        return current_loss / count
end

print(testingSize)
function eval(batch_size)

        local count = 0
        batch_size = batch_size or 200

        local lstidWordOut = {}
        local lstidWordDest = {}
        local idxWord = 1

        true_prob = {}
        false_prob = {}
        data_prob = {}

        -- init true and false count --
        for i=1,nCountTopic do
                true_prob[i] = 0
                false_prob[i] = 0
                data_prob[i] = 0
        end

        for i = 0, testingSize, batch_size do
                if i >= testingSize then
                        break
                end

                local size = math.min(i + batch_size, testingSize) - i

                --                local inputs = data_test[{{i+1,i+size}}]
                --                local targets = labels_test[{{i+1,i+size}}]:long()

                local inputs = torch.Tensor(size, windowSize)
                local targets = torch.Tensor(size)


                for t = 1, size do

                        local input = data_test[i+t]
                        local target = labels_test[i+t]
                        -- if target == 0 then target = 10 end
                        inputs[t] = input
                        targets[t] = target
                end

                local outputs = model:forward(inputs)
                local _, indices = torch.max(outputs, 2)

                guessed_right = 0
                for j=1, indices:size()[1] do

                        label = targets[j]
                        data_prob[label] = data_prob[label]+1

                        -- tong hop ket qua
                        lstidWordOut[idxWord] = indices[j][1]
                        lstidWordDest[idxWord] = targets[j]
                        idxWord = idxWord +1

                        if indices[j][1] == targets[j] then
                                guessed_right = guessed_right + 1
                                true_prob[label] = true_prob[label] + 1
                        else
                                false_prob[label] = false_prob[label] + 1
                        end
                end

                count = count + guessed_right
        end

        -- Thong ke lai ket qua
        local tblResultTrue = torch.LongTensor(nCountTopic):fill(0)
        local tblResultOut = torch.LongTensor(nCountTopic):fill(0)
        local tblNERDataset = torch.LongTensor(nCountTopic):fill(0)
        local sumTrue, sumOut, sumData = 0,0,0


        local tblResultTrue2 = torch.LongTensor(nCountTopic):fill(0)
        local tblResultOut2 = torch.LongTensor(nCountTopic):fill(0)
        local tblNERDataset2 = torch.LongTensor(nCountTopic):fill(0)
        local sumTrue2, sumOut2, sumData2 = 0,0,0

        local nSizeInputs = #lstidWordDest

        for idxWord = 1, nSizeInputs do

                local idNEROut, idNERDist -- ket qua gan nhan cho tu
                idNERDist = lstidWordDest[idxWord]
                idNEROut  = lstidWordOut[idxWord]


                if idNEROut == idNERDist then
                        tblResultTrue [idNERDist] = tblResultTrue[idNERDist]+1
                end
                tblResultOut[idNEROut] =  tblResultOut[idNEROut] + 1
                tblNERDataset[idNERDist] = tblNERDataset[idNERDist] + 1

                -- tong hop ket qua Theo nhan ngu nghia tung cau vao tap ket qua 2
                if(idNEROut == 1 or idNEROut % 2 == 0) then
                        tblResultOut2[idNEROut] =  tblResultOut2[idNEROut] + 1
                end

                if(idNERDist == 1 or idNERDist % 2 == 0) then
                        tblNERDataset2[idNERDist] = tblNERDataset2[idNERDist] + 1
                end

                if(idNERDist == idNEROut) then
                        if(idNEROut == 1) then
                                tblResultTrue2[idNEROut] =  tblResultTrue2[idNEROut] + 1
                        else
                                -- Neu la nhan I-
                                if(idNEROut % 2 == 1) then goto _CONTINUE_NEXT_WORD_ end

                                -- Neu nhan B- la word cuoi cung trong cau
                                if(idxWord == nSizeInputs) then
                                        tblResultTrue2[idNEROut] =  tblResultTrue2[idNEROut] + 1
                                        goto _CONTINUE_NEXT_WORD_
                                end

                                -- Kiem tra cac nhan I- phia sau
                                local j = 0
                                for j = idxWord+1, nSizeInputs do

                                        local idNEROutNext = lstidWordOut[j]
                                        local idNERDistNext = lstidWordDest[j]

                                        -- Neu cac tu phia sau trong Out va Dist deu khong phai la I- => +1
                                        if(idNERDistNext ~= (idNEROut+1) and idNEROutNext ~= (idNEROut+1)) then
                                                tblResultTrue2[idNEROut] =  tblResultTrue2[idNEROut] + 1
                                                break
                                        end

                                        -- neu 1 trong 2 tu phia sau == IdNEROut + 1 nhung khac nhau => +0
                                        if(idNERDistNext ~= idNEROutNext) then
                                                break
                                        end

                                        -- Neu 2 tu (Out va Dist) = nhau va = idNEROut + 1 va la word cuoi cung
                                        -- trong cau => +1
                                        if(j == nSizeInputs) then
                                                tblResultTrue2[idNEROut] =  tblResultTrue2[idNEROut] + 1
                                        end

                                end

                        end
                end
                ::_CONTINUE_NEXT_WORD_::
        end

        -- Ket qua tong hop chung tat ca cac nhan ngoai tru nhan O
        for i = 2, nCountTopic do
                sumData = sumData + tblNERDataset[i]
                sumData2 = sumData2+tblNERDataset2[i]

                sumOut = sumOut + tblResultOut[i]
                sumOut2 = sumOut2 + tblResultOut2[i]

                sumTrue = sumTrue + tblResultTrue[i]
                sumTrue2 = sumTrue2 + tblResultTrue2[i]
        end

        print '................................................................'
        print 'Khong ghep cap B- I- : '
        print 'So tu gan nhan dung / So tu duoc gan nhan / So tu trong dataset '
        print (string.format('Precission = %6.2f, Recall = %6.2f ',
                sumTrue*100.0/sumOut, sumTrue*100.0/sumData))
        print '................................................................'
        for i=1, nCountTopic do
                print(string.format("%6d, %6d, %6d,  P = %6.2f, R = %6.2f",
                        tblResultTrue[i], tblResultOut[i], tblNERDataset[i],
                        tblResultTrue[i]*100.00/tblResultOut[i],
                        tblResultTrue[i]*100.00/tblNERDataset[i]
                ))
        end


        print '................................................................'
        print 'Ghep cap BI- : '
        print 'So tu gan nhan dung / So tu duoc gan nhan / So tu trong dataset '
        print (string.format('Precission = %6.2f, Recall = %6.2f ',
                sumTrue2*100.0/sumOut2, sumTrue2*100.0/sumData2))
        print '................................................................'
        for i=1, nCountTopic do
                print(string.format("%6d, %6d, %6d,  P = %6.2f, R = %6.2f",
                        tblResultTrue2[i], tblResultOut2[i], tblNERDataset2[i],
                        tblResultTrue2[i]*100.00/tblResultOut2[i],
                        tblResultTrue2[i]*100.00/tblNERDataset2[i]
                ))
        end

        print '................................................................'

        ::_EXIT_FUNCTION_::

        return count/testingSize, true_prob, false_prob, data_prob
end

max_iters = 30
do

        local last_accuracy = 0
        local decreasing = 0
        local threshold = 1 -- how many deacreasing epochs we allow
        for iter = 1, max_iters do

                print(string.format("-----start train [dataset %d - iter %d/%d]------", iDataSet, iter, max_iters))
                local loss = step(20)
                print(string.format('Epoch: %d Current loss: %4f', iter, loss))


                print(string.format("-----start test [dataset %d - iter %d/%d]------", iDataSet, iter, max_iters))
                local accuracy, true_prob, false_prob = eval(20)
                print(string.format('Accuracy on the validation set: %4f', accuracy))
                for x, y in pairs(true_prob) do
                        print(string.format('Count label %d number of true=%d, number of false=%d, accuracy=%4f', x, y, false_prob[x], y/(y+false_prob[x])))
                end


                if accuracy < last_accuracy then
                        if decreasing > threshold then break end
                        decreasing = decreasing + 1
                else
                        decreasing = 0
                end
                last_accuracy = accuracy
        end

end

