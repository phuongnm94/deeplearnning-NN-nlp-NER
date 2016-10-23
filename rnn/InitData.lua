require('LibsExtend')
require('nn')
nninit = require('nninit')


---
-- Doc ma tran khoi tao sinh boi bo word to vect
--
-- @function [parent=#global] loadMatrixWordToVect(sPathFileDict)
-- @param sPathFileDict Duong dan file ma tran
-- @return pairWordIds: ex: {... {toi, 5}, {an, 9}, ...}
-- @return Matran trong so ex: {... 5:{0.1, 0.2, ...}, 9:{0.3,0.4, ...}, ...}
--
function loadMatrixWordToVect(sPathFileDict)

        -- create file
        local pairWordIds, mtWordVector, pairIdWords

        -- Doc ma tran vector cua cac tu
        pairWordIds, mtWordVector,pairIdWords = matrixExtractFrom(sPathFileDict)

        ::_EXIT_FUNCTION::
        return pairWordIds, mtWordVector, pairIdWords
end

---
-- Doc du lieu  nhan - va gia tri cua cac nhan NER
--
-- @function [parent=#global] getNERIds()
--
-- @return NERIds
--
function getNERIds ()

        local NERIds= {}
        local NERLabels= {}

        --        NERIds["O"] = 1
        --
        --        NERIds["B-LOC"] = 2
        --        NERIds["I-LOC"] = 3
        --
        --        NERIds["B-ORG"] = 4
        --        NERIds["I-ORG"] = 5
        --
        --        NERIds["B-PER"] = 6
        --        NERIds["I-PER"] = 7
        --
        --        NERIds["B-TOUR"] = 8
        --        NERIds["I-TOUR"] = 9
        --
        --        NERIds["B-PRO"] = 10
        --        NERIds["I-PRO"] = 11


        -- Data on vc corp
        NERIds["O"] = 1
        NERIds["O       "] = 1
        NERIds["O\9"] = 1
        NERIds["OO"] = 1
        NERIds["B-TOOUR"] = 1
        NERIds["B-TOUR"] = 1
        NERIds["I-TOUR"] = 1
        NERIds["BO"] = 1
        NERLabels[1] = "O"


        NERIds["B-LOC"] = 2
        NERIds["B-LOCO"] = 2
        NERIds["I-LOC"] = 3
        NERLabels[2] = "B-LOC"
        NERLabels[3] = "I-LOC"

        NERIds["B-ORG"] = 4
        NERIds["I-ORG"] = 5
        NERIds["I-ORGO"] = 5
        NERLabels[4] = "B-ORG"
        NERLabels[5] = "I-ORG"


        NERIds["B-PRO"] = 6
        NERIds["B-Pro"] = 6
        NERIds["I-PRO"] = 7
        NERLabels[6] = "B-PRO"
        NERLabels[7] = "I-PRO"

        NERIds["B-PER"] = 8
        NERIds["I-PER"] = 9
        NERLabels[8] = "B-PER"
        NERLabels[9] = "I-PER"


        return NERIds, NERLabels
end

---
-- mo rong bo du lieu random cho cac tu khong co trong tu dien
--
-- @function [parent=#global] appendRandomDataWeightMatrix(matrix, size)
--
-- @param matrix - doubleTensor matran duoc thay doi
-- @param size so ban ghi them vao
-- @param dim chieu mo rong
--
-- @return matrix sau khi duoc them 'dim[size]' ngau nhien theo phan phoi Grotot
function appendRandomDataWeightMatrix(matrix, size, dim)

        local sizeMt = matrix:size()
        sizeMt[1] = size

        local tmpNetLinear = nn.Linear(sizeMt[2], sizeMt[1])
        tmpNetLinear:init('weight', nninit.xavier)

        matrix = torch.cat(matrix, tmpNetLinear.weight, dim)
        return matrix
end


---
-- Cai dat bo du lieu random cho cac tu khong co trong tu dien
--
-- @function [parent=#global] initRandomDataWeightMatrix(x, y)
--
-- @param x chieu 1 - dims 1
-- @param y chieu 2 - dims 2
--
-- @return matrix sau khi duoc them 'dim[x, y]' ngau nhien theo phan phoi Grotot
--         matrix weight[x, y] by Grotot distribute
function initRandomDataWeightMatrix(x, y)

        local tmpNetLinear = nn.Linear(y, x)
        tmpNetLinear:init('weight', nninit.xavier)

        return tmpNetLinear.weight
end

---
-- Chia tap du lieu thanh cac bo data kich thuoc < sizeSplit
-- tap du lieu tra ve se gom cac phan tu - moi phan tu la 1 tap cac cau co
-- so tu bang nhau = 1 bo du lieu cho 1 lan hoc
--
-- @function [parent=#global] splitDataSet(inputs, targets, sizeSplit)
--
-- @param inputs tap du lieu cau dau vao
-- @param targets tap du lieu cau sau khi dc gan nhan
-- @param sizeSentencesInfo thong tin kich thuoc cua cau
-- @param sizeSplit so luong cau toi da cho moi lan hoc
--
-- @return inputs: Cac cau : {... {1, 5, 9, 8, 22,25}, {2,7,8,1,9}, ...}
-- @return targets: NER tuong ung : {... {1, 1, 1, 1, 1, 2}, {3,4,1,1,1}, ...}
--
function splitDataSet2(inputs, targets,features, sizeSentencesInfo, sizeSplit)

        -- load data file
        local inputsSplit, targetsSplit = {}, {}
        local mtInputsSplit, mtTargetsSplit, mtFeaturesSplit = {}, {}, {}
        local idxMtInputsSplit, nSizeMtInputsSplit = 1, 0
        local idxInputsSplit = 1


        for sizeSentence, lstIdSentence in pairs(sizeSentencesInfo) do

                local sizeList, idxList = #lstIdSentence, 1

                if(sizeList <= sizeSplit) then
                        inputsSplit[idxInputsSplit] = lstIdSentence
                        idxInputsSplit = idxInputsSplit + 1
                else
                        local nCountBatch = math.floor(sizeList / sizeSplit)

                        for i = 1, nCountBatch do
                                inputsSplit[idxInputsSplit] = {}

                                for j = 1, sizeSplit  do
                                        inputsSplit[idxInputsSplit][j] = lstIdSentence[idxList]
                                        idxList = idxList + 1
                                end

                                idxInputsSplit = idxInputsSplit + 1
                        end

                        inputsSplit[idxInputsSplit] = {}

                        local i = 1
                        for j = idxList, sizeList do

                                inputsSplit[idxInputsSplit][i] = lstIdSentence[j]
                                i = i + 1
                        end

                        if(i > 1) then
                                idxInputsSplit = idxInputsSplit  + 1
                        end

                end
                --                break

        end
        --        print(inputsSplit)

        nSizeMtInputsSplit = #inputsSplit
        print('nSizeMtInputsSplit : ' .. nSizeMtInputsSplit)
        for idxMtInputsSplit = 1, nSizeMtInputsSplit do

                local lstIdSentence = inputsSplit[idxMtInputsSplit]     -- list id in batch input
                local nSizeLstIdSentence = #lstIdSentence               -- kich thuoc cua 1 vector

                local nSizeOneSenctence = #(inputs[lstIdSentence[1]])     -- so luong cac vector

                mtInputsSplit[idxMtInputsSplit] = {}                    -- 1 batch phantu input ~ 20
                mtTargetsSplit[idxMtInputsSplit] = {}
                mtFeaturesSplit[idxMtInputsSplit] = {}

                for i = 1, nSizeOneSenctence do
                        mtInputsSplit[idxMtInputsSplit][i] = {}--torch.LongTensor(nSizeLstIdSentence)
                        mtTargetsSplit[idxMtInputsSplit][i] = {}--torch.LongTensor(nSizeLstIdSentence)

                        for j=1, nSizeLstIdSentence do
                                mtInputsSplit[idxMtInputsSplit][i][j] = inputs[lstIdSentence[j]][i]
                                mtTargetsSplit[idxMtInputsSplit][i][j] = targets[lstIdSentence[j]][i]
                        end

                end

                if(features ~= nil) then
                        for j=1, nSizeLstIdSentence do
                                mtFeaturesSplit[idxMtInputsSplit][j] = features[lstIdSentence[j]]
                        end
                end

        end

        g_inputsBatchIds = inputsSplit

        ::_EXIT_FUNCTION::
        return  mtInputsSplit, mtTargetsSplit, mtFeaturesSplit
end


---
-- Chia tap du lieu thanh cac bo data kich thuoc < sizeSplit
-- tap du lieu tra ve se gom cac phan tu - moi phan tu la 1 tap cac cau co
-- so tu bang nhau = 1 bo du lieu cho 1 lan hoc
--
-- @function [parent=#global] splitDataSet(inputs, targets, sizeSplit)
--
-- @param inputs tap du lieu cau dau vao
-- @param targets tap du lieu cau sau khi dc gan nhan
-- @param sizeSentencesInfo thong tin kich thuoc cua cau
-- @param sizeSplit so luong cau toi da cho moi lan hoc
--
-- @return inputs: Cac cau : {... {1, 5, 9, 8, 22,25}, {2,7,8,1,9}, ...}
-- @return targets: NER tuong ung : {... {1, 1, 1, 1, 1, 2}, {3,4,1,1,1}, ...}
--
function splitDataSet(inputs, targets,features, sizeSentencesInfo, sizeSplit)

        -- load data file
        local inputsSplit, targetsSplit = {}, {}
        local mtInputsSplit, mtTargetsSplit, mtFeaturesSplit = {}, {}, {}
        local idxMtInputsSplit, nSizeMtInputsSplit = 1, 0
        local idxInputsSplit = 1

        for sizeSentence, lstIdSentence in pairs(sizeSentencesInfo) do

                local sizeList, idxList = #lstIdSentence, 1

                if(sizeList <= sizeSplit) then
                        inputsSplit[idxInputsSplit] = lstIdSentence
                        idxInputsSplit = idxInputsSplit + 1
                else
                        local nCountBatch = math.floor(sizeList / sizeSplit)

                        for i = 1, nCountBatch do
                                inputsSplit[idxInputsSplit] = {}

                                for j = 1, sizeSplit  do
                                        inputsSplit[idxInputsSplit][j] = lstIdSentence[idxList]
                                        idxList = idxList + 1
                                end

                                idxInputsSplit = idxInputsSplit + 1
                        end

                        inputsSplit[idxInputsSplit] = {}

                        local i = 1
                        for j = idxList, sizeList do

                                inputsSplit[idxInputsSplit][i] = lstIdSentence[j]
                                i = i + 1
                        end

                        if(i > 1) then
                                idxInputsSplit = idxInputsSplit  + 1
                        end

                end
                --                break

        end
        --        print(inputsSplit)

        nSizeMtInputsSplit = #inputsSplit
        print('nSizeMtInputsSplit : ' .. nSizeMtInputsSplit)
        for idxMtInputsSplit = 1, nSizeMtInputsSplit do

                local lstIdSentence = inputsSplit[idxMtInputsSplit]     -- list id in batch input
                local nSizeLstIdSentence = #lstIdSentence               -- kich thuoc cua 1 vector

                local nSizeOneSenctence = #(inputs[lstIdSentence[1]])     -- so luong cac vector

                mtInputsSplit[idxMtInputsSplit] = {}                    -- 1 batch phantu input ~ 20
                mtTargetsSplit[idxMtInputsSplit] = {}
                mtFeaturesSplit[idxMtInputsSplit] = {}

                for j=1, nSizeLstIdSentence do

                        mtInputsSplit[idxMtInputsSplit][j] = inputs[lstIdSentence[j]]
                        mtTargetsSplit[idxMtInputsSplit][j] = targets[lstIdSentence[j]]
                end

                if(features ~= nil) then
                        for j=1, nSizeLstIdSentence do
                                mtFeaturesSplit[idxMtInputsSplit][j] = features[lstIdSentence[j]]
                        end
                end

        end

        ::_EXIT_FUNCTION::
        return  mtInputsSplit, mtTargetsSplit, mtFeaturesSplit
end


---
-- Chia tap du lieu thanh cac bo data kich thuoc <= sizeSplit
-- tap du lieu tra ve se gom cac phan tu - moi phan tu la 1 tap cac cau co
-- so tu bang nhau = 1 bo du lieu cho 1 lan hoc
--
-- @function [parent=#global] splitDataSet(inputs, targets, sizeSplit)
--
-- @param inputs tap du lieu cau dau vao
-- @param targets tap du lieu cau sau khi dc gan nhan
-- @param sizeSentencesInfo thong tin kich thuoc cua cau
-- @param sizeSplit so luong cau toi da cho moi lan hoc
--
-- @return inputs: Cac cau : {... {1, 5, 9, 8, 22,25}, {2,7,8,1,9}, ...}
-- @return targets: NER tuong ung : {... {1, 1, 1, 1, 1, 2}, {3,4,1,1,1}, ...}
--
function splitDataSetUseMaskZeroPadding(inputs, targets,features, sizeSplit)

        -- load data file
        local mtInputsSplit, mtTargetsSplit, mtFeaturesSplit = {}, {}, {}
        local tmpBatchSentence, tmpBatchSentenceDest, tmpBatchSentenceFeature = {},{}, {}
        local maxWordInSentence = 0

        for idxSentence, sentence in pairs(inputs) do

                -- them tung cau vao current batch
                tmpBatchSentence[#tmpBatchSentence+1] = sentence
                tmpBatchSentenceDest[#tmpBatchSentenceDest+1] = targets[idxSentence]
                if(features ~= nil) then
                        tmpBatchSentenceFeature[#tmpBatchSentenceFeature+1] = features[idxSentence]
                end

                maxWordInSentence = math.max(maxWordInSentence, #sentence)

                -- them 1 batch sentence vao table tra ve
                if (#tmpBatchSentence == sizeSplit or idxSentence == #inputs) then

                        -- add padding cho tung cau
                        for idx, sentence in pairs(tmpBatchSentence) do
                                local nCountPaddingAdd = maxWordInSentence - #sentence
                                if nCountPaddingAdd == 0 then goto CONTINUE end

                                local tblPadding =  torch.totable(torch.zeros(nCountPaddingAdd))
                                tmpBatchSentence[idx] = tableEx(sentence):append(tblPadding)
                                tmpBatchSentenceDest[idx] = tableEx(tmpBatchSentenceDest[idx]):append(tblPadding)

                                ::CONTINUE::
                        end

                        -- them batch sentence vao table
                        mtInputsSplit[#mtInputsSplit+1] = tmpBatchSentence
                        mtTargetsSplit[#mtTargetsSplit+1] = tmpBatchSentenceDest
                        if(features ~= nil) then
                                mtFeaturesSplit[#mtFeaturesSplit+1] = tmpBatchSentenceFeature
                        end

                        tmpBatchSentence, tmpBatchSentenceDest, tmpBatchSentenceFeature = {},{}, {}
                        maxWordInSentence = 0
                end

        end

        ::_EXIT_FUNCTION::
        return  mtInputsSplit, mtTargetsSplit, mtFeaturesSplit
end

---
-- Thong ke lai ti le trainTest tung tap dataset cho tung chu de
--
-- @function [parent=#global] showRateDatasetTrainTest(datasetNew)
-- @param  datasetNew tap du lieu da duoc chia lam 10 phan
function showRateDatasetTrainTest (datasetNew)

        local iDatasetTest = 1

        for iDatasetTest =1, 10 do
                print(string.format("Dataset %d", iDatasetTest))

                local test  = {}
                local train = tableEx({})

                local countTest  = torch.Tensor(g_nCountLabel):fill(0)
                local countTrain  = torch.Tensor(g_nCountLabel):fill(0)

                for idxDataset, subDataset in pairs(datasetNew) do
                        if idxDataset == iDatasetTest then
                                test = subDataset
                        else
                                train = train:append(subDataset)
                        end
                end

                for i, sentence in pairs(test) do
                        for iWord , idLabel in pairs (sentence) do
                                countTest[idLabel] =  countTest[idLabel] +1
                        end
                end

                for i, sentence in pairs(train) do
                        for iWord , idLabel in pairs (sentence) do
                                countTrain[idLabel] =  countTrain[idLabel] +1
                        end
                end

                for i=1, g_nCountLabel do
                        print (string.format("test = %d, train = %d, rate = %6.2f ",
                                countTest[i],
                                countTrain[i],
                                countTest[i]*100.0/(countTrain[i]+countTest[i])
                        ))
                end
        end
end

---
-- Danh lai chi so cac cau sao cho moi tap du lieu (10 sub dataset) chua so
-- luong cac tu thuoc cac nhan deu nhau / ReIndex sentence to balance label's
-- countWord each sub dataset
--
-- @function [parent=#global] reIndexDataset(dataset)
-- @param  dataset tap du lieu targets input / targetsInput set
function reIndexDataset(inputs, targets, features)

        local inputsNew, targetsNew, featuresNew = {}, {}, {}
        local infoLabelToSentence = {}
        local infoSizeLabelSubDataset = torch.Tensor(g_nCountLabel):fill(0)
        local mtDatasetIsUsed = torch.Tensor(#targets):fill(0)
        
        -- khoi tao tap thong tin cho tung nhan
        for i =1, g_nCountLabel do infoLabelToSentence [i] = {} end

        -- Khoi tao bo chua du lieu sub dataset
        for i =1, 10 do 
               inputsNew[i], targetsNew[i], featuresNew[i] = {}, {}, {} 
        end

        -- Thong ke lai { idLabel -> {idSentence,.}, idLabel -> {idSentence,..} }
        for idxSentence, sentence in pairs(targets) do
                local tmpSentenceWordId = {}

                -- tim kiem tap nhan trong cau
                for _, wordLabelId in pairs(sentence) do
                        if (wordLabelId > 0) then
                                tmpSentenceWordId[wordLabelId] = true
                        end
                end

                -- dua tung cau vao tap nhan -> {idSentence} tuong ung
                for wordLabelId, _ in pairs(tmpSentenceWordId) do
                        if (wordLabelId > 0) then
                                table.insert(infoLabelToSentence[wordLabelId], idxSentence)
                        end
                end
        end

        -- Tinh lai so phan tu cua tung tap nhan
        for idLabel, dataset in pairs(infoLabelToSentence) do
                infoSizeLabelSubDataset[idLabel] = math.ceil(#dataset / 10)
        end

        -- gan lai chi so vao tap du lieu moi
        for idLabel = g_nCountLabel, 1, -1 do

                local setIdSentence = infoLabelToSentence[idLabel]

                for idx, idSentence in pairs(setIdSentence) do

                        -- Kiem tra xem cau nay da duoc dung hay chua
                        if(mtDatasetIsUsed[idSentence] == 1) then goto CONTINUE end

                        -- Ghi lai cau vao vi tri sub dataset moi
                        local idxSubDataset = math.ceil(idx/infoSizeLabelSubDataset[idLabel])
                        table.insert(targetsNew[idxSubDataset], targets[idSentence])
                        table.insert(inputsNew[idxSubDataset], inputs[idSentence])
                        if(features~= nil) then 
                                table.insert(featuresNew[idxSubDataset], features[idSentence])
                        end
                        mtDatasetIsUsed[idSentence] = 1

                        ::CONTINUE::
                end
        end
        
        --showRateDatasetTrainTest(targetsNew)
        return inputsNew, targetsNew, featuresNew

end

---
-- Doc du lieu  tu tap data Set
--
-- @function [parent=#global] loadDataSet(sPathFileDataSet, pairWordIds, NERIds)
--
-- @param sPathFileDataSet Duong dan file DataSet
-- @param pairWordIds bang chua cac tu va id cua cac tu do
-- @param NERIds nhan phan loai cac loai NER
--
-- @return inputs: Cac cau : {... {1, 5, 9, 8, 22,25}, {2,7,8,1,9}, ...}
-- @return targets: NER tuong ung : {... {1, 1, 1, 1, 1, 2}, {3,4,1,1,1}, ...}
--
function loadDataSet(sPathFileDataSet, pairWordIds, NERIds,nLastIdxMtWordVector, sizeAppendDict)

        local inputs, targets, features, inputsNew, targetsNew, featuresNew
        DataSet, DataSizeInfo, DataSetGroup = {}, {}, {}

        -- Doc du lieu file tu dien 
        inputs, targets, g_sizeSentencesInfo, features=
                getDataSentenceFrom2(sPathFileDataSet,pairWordIds,NERIds,nLastIdxMtWordVector, sizeAppendDict)
        assert(#inputs == #targets, 'Input vs Target is not same size')

        if(g_isReparseBalanceData == true) then 
        
                DataSet["inputsTrain"] = tableEx({})
                DataSet["targetsTrain"] = tableEx({})
                DataSet["featuresTrain"] = tableEx({})
                inputsNew, targetsNew, featuresNew = reIndexDataset(inputs, targets, features)
                for idxDataset=1, 10 do
                        if idxDataset == g_iDataset then
                                DataSet["inputsTest"] = inputsNew[idxDataset]
                                DataSet["targetsTest"] = targetsNew[idxDataset]
                                if g_isUseFeatureWord then 
                                        DataSet["featuresTest"] = featuresNew[idxDataset]
                                end
                        else    
                                DataSet["inputsTrain"] = DataSet["inputsTrain"]:append(inputsNew[idxDataset])
                                DataSet["targetsTrain"] = DataSet["targetsTrain"]:append(targetsNew[idxDataset])
                                if g_isUseFeatureWord then
                                        DataSet["featuresTrain"] = DataSet["featuresTrain"]:append(featuresNew[idxDataset])
                                end
                        end
                end
                if (g_isUseFeatureWord and g_iModelTest ==  2) then 
                        GenerateLinguiticsFeatureWeight(DataSet)
                end 
        else 
                local nCountInputs = #inputs
                local nCountTestSet = math.max(1,math.ceil((1 - g_trainRate) * nCountInputs));
                local nIndexTestStart = math.min(
                        math.max(1, math.ceil((g_iDataset-1)*(1-g_trainRate)*nCountInputs)),
                        nCountInputs
                )
                local nIndexTestEnd = math.min(nCountInputs, nCountTestSet + nIndexTestStart )
                
                print(string.format('Dataset test [%d/%d] : Cau %d - Cau %d', g_iDataset, 10, nIndexTestStart, nIndexTestEnd))
        
                if(_VERSION == "Lua 5.2") then
                        DataSet["inputsTest"] = {table.unpack(inputs,nIndexTestStart,nIndexTestEnd)}
                        DataSet["inputsTrain"] = (tableEx({table.unpack(inputs,1,nIndexTestStart-1)}))
                                :append({table.unpack(inputs,nIndexTestEnd+1,nCountInputs)})
        
                        DataSet["targetsTest"] = {table.unpack(targets,nIndexTestStart,nIndexTestEnd)}
                        DataSet["targetsTrain"] =  (tableEx({table.unpack(targets,1,nIndexTestStart-1)}))
                                :append({table.unpack(targets,nIndexTestEnd+1,nCountInputs)})
                else
                        DataSet["inputsTest"] = tableUnpackExtend(inputs,nIndexTestStart,nIndexTestEnd)
                        DataSet["inputsTrain"] = (tableEx(tableUnpackExtend(inputs,1,nIndexTestStart-1)))
                                :append(tableUnpackExtend(inputs,nIndexTestEnd+1,nCountInputs))
        
                        DataSet["targetsTest"] = tableUnpackExtend(targets,nIndexTestStart,nIndexTestEnd)
                        DataSet["targetsTrain"] = (tableEx(tableUnpackExtend(targets,1,nIndexTestStart-1)))
                                :append(tableUnpackExtend(targets,nIndexTestEnd+1,nCountInputs))
                end
        end

        DataSizeInfo["inputsTest"] = groupSentenceSameSize(DataSet["inputsTest"])
        DataSizeInfo["inputsTrain"] = groupSentenceSameSize(DataSet["inputsTrain"])


        -- Chia bo du lieu input[i] = matrix[batchSize x countSentence x countWord]
        -- split data to input[i] = matrix[batchSize x countSentence x countWord]
        if (bIsRunInParalell == true) then

                --inputsParallel,targetsParalel, featuresParallel =
                --        splitDataSet(inputs,targets, features, g_sizeSentencesInfo, g_batchSentenceSize)

                if g_isUseMaskZeroPadding == true then

                        DataSetGroup["inputsTrain"], DataSetGroup["targetsTrain"], DataSetGroup["featuresTrain"]
                        =  splitDataSetUseMaskZeroPadding(
                                DataSet["inputsTrain"] ,DataSet["targetsTrain"] ,
                                DataSet["featuresTrain"], g_batchSentenceSize)

                        DataSetGroup["inputsTest"], DataSetGroup["targetsTest"], DataSetGroup["featuresTest"]
                        =  splitDataSetUseMaskZeroPadding(
                                DataSet["inputsTest"] ,DataSet["targetsTest"] ,
                                DataSet["featuresTest"], g_batchSentenceSize)

                        goto _END_SPLIT_DATA
                end
                
                DataSetGroup["inputsTrain"], DataSetGroup["targetsTrain"], DataSetGroup["featuresTrain"]
                =  splitDataSet(DataSet["inputsTrain"] ,DataSet["targetsTrain"] ,
                        DataSet["featuresTrain"] , DataSizeInfo["inputsTrain"], g_batchSentenceSize)

                DataSetGroup["inputsTest"], DataSetGroup["targetsTest"], DataSetGroup["featuresTest"]
                =  splitDataSet(DataSet["inputsTest"] ,DataSet["targetsTest"] ,
                        DataSet["featuresTest"] , DataSizeInfo["inputsTest"], g_batchSentenceSize)
                ::_END_SPLIT_DATA::
        end

        ::_EXIT_FUNCTION::
        return  inputs, targets, features
end


---
-- Sinh vector input tuong ung cac tu(word) trong cau
--
-- @function [parent=#global]  generateInputMatrix(inputSentence, sizeWordVector)
-- @param inputSentence {idWords} ds id cac tu trong cau
-- @param sizeWordVector number kich thuoc vector cua 1 tu
--
-- @return vectorWord vectorWord[i == wordId] = 1 and vectorWord[id#wordId] = 0
function generateSentenceInputMatrix(inputSentence, sizeWordVector)

        local sentenceInputMatrix = {}
        for k, v in pairs(inputSentence) do
                sentenceInputMatrix[k] = torch.Tensor(sizeWordVector):fill(0)
                sentenceInputMatrix[k][v] = 1
        end

        return sentenceInputMatrix
end

---
-- Cai dat du lieu dataset : trainning and testing
--
-- @function [parent=#global]  InitData()
function GetRateTrainingEachClass(dataset, nIndexStart, nCountElement)

        local rateClassRet = {}
        local mtRateRet = nil
        local nCountClass = 0
        local sumAllWord = 0
        local nSizeDataset = #dataset

        if(nCountElement == nil ) then

                nCountElement = #dataset --math.ceil((#dataset*0.9) + 1)
        end

        if(nIndexStart == nil ) then

                nIndexStart = 1
        end

        for i = 1, g_nCountLabel do
                rateClassRet[i] = 0
        end

        if(bIsRunInParalell == nil or bIsRunInParalell == false) then

                -- voi bo du lieu thong thuong
                -- moi phan tu trong targets la 1 cau
                for idxSentence = nIndexStart, nCountElement+nIndexStart-1 do

                        -- tinh toan lai gia tri chi so cau
                        local idxSentenceReally= nil
                        if (idxSentence%nSizeDataset == 0) then
                                idxSentenceReally = nSizeDataset
                        else
                                idxSentenceReally = idxSentence%nSizeDataset
                        end

                        local sentence = dataset[idxSentenceReally]
                        for k, v in pairs(sentence) do
                                if(rateClassRet[v] == nil)then
                                        rateClassRet[v] = 1
                                else
                                        rateClassRet[v] = rateClassRet[v]+1
                                end

                        end
                end
        else
                -- voi bo du lieu song song
                -- moi phan tu trong targets la 1 tap nhieu cau
                local batchSentence = nil
                local idWord = nil
                for idxDataset = nIndexStart, nIndexStart+nCountElement-1 do

                        -- tinh toan lai gia tri chi so cau
                        local idxDatasetReally= nil
                        if (idxDataset%nSizeDataset == 0) then
                                idxDatasetReally = nSizeDataset
                        else
                                idxDatasetReally = idxDataset%nSizeDataset
                        end

                        batchSentence = dataset[idxDatasetReally]
                        local nCountWordOneSentence, nCountSentence = #batchSentence, #(batchSentence[1])

                        for idxWord= 1, nCountWordOneSentence do
                                for idxSentence = 1, nCountSentence do
                                        idWord = batchSentence[idxWord][idxSentence]
                                        if(rateClassRet[idWord] == nil)then
                                                rateClassRet[idWord] = 1
                                        else
                                                rateClassRet[idWord] = rateClassRet[idWord]+1
                                        end
                                end
                        end

                end

        end

        -- skip padding value: key = 0
        rateClassRet[0] = nil

        for k, v in pairs(rateClassRet) do

                sumAllWord = sumAllWord + v
                nCountClass = nCountClass + 1

        end

        mtRateRet = torch.DoubleTensor(nCountClass):fill(0.00)
        for idx = 1, nCountClass do
                if(rateClassRet[idx] ~= 0) then
                        mtRateRet[idx] = sumAllWord*1.00 / rateClassRet[idx]
                else
                        mtRateRet[idx] = 0
                end
        end
        return mtRateRet, rateClassRet
end


---
-- Sinh ma tran trong so = distinct(featuresMatrix) -> sd lookup table to weight
--
-- @function [parent=#global]  GenerateLinguiticsFeatureWeight(data)
function GenerateLinguiticsFeatureWeight(data)
        
        local mtWeightFeature = tableEx({})
        local mtWeightFeatureReverse = tableEx({})
        local idFeature, keyReverse
        for idxSentence, sentence in pairs(data["featuresTrain"]) do 
                for idxWord, vtWordFeature in pairs(sentence)do 
                
                        -- trich xuat key cua features = 10010100101
                        keyReverse = table.concat(vtWordFeature,'')
                        
                        -- kiem tra key nay co trong ma tran features hay chua
                        idFeature = mtWeightFeatureReverse[keyReverse]
                        if (idFeature == nil)then 
                                idFeature = #mtWeightFeature+1
                                mtWeightFeature[idFeature] = vtWordFeature
                                mtWeightFeatureReverse[keyReverse] = idFeature
                        end 
                        sentence[idxWord] = idFeature
                end       
        end
        for idxSentence, sentence in pairs(data["featuresTest"]) do 
                for idxWord, vtWordFeature in pairs(sentence)do 
                        
                        -- trich xuat key cua features = 10010100101
                        keyReverse = table.concat(vtWordFeature,'')
                        
                        -- kiem tra key nay co trong ma tran features hay chua
                        idFeature = mtWeightFeatureReverse[keyReverse]
                        if (idFeature == nil)then
                                idFeature = #mtWeightFeature+1
                                mtWeightFeature[idFeature] = vtWordFeature
                                mtWeightFeatureReverse[keyReverse] = idFeature
                        end 
                        sentence[idxWord] = idFeature
                        
                        
                        
                end       
        end
        data['featuresWeight']= torch.Tensor(mtWeightFeature)
        g_nFeatureDims = #mtWeightFeature[1]
        g_nFeatureSize = #mtWeightFeature
        print (string.format('size Weight Linguitics Features = [%dx%d]', g_nFeatureSize, g_nFeatureDims))
end



---
-- Cai dat du lieu dataset : trainning and testing
--
-- @function [parent=#global]  InitData(sDictName, sDataSetName, sPathFileFeature)
-- @param sDictName ten file tu dien
-- @param sDataSetName ten file dataset chua cac cau da duoc gan nhan cho tap hoc
-- @param sPathFileFeature ten file feature chua cac dac trung ngon ngu
function InitData(sDictName, sDataSetName)

        local bTestDebug = false
        if(bTestDebug == true) then

                pairWordIds, mtWordVector, pairIdWords = loadMatrixWordToVect('te.txt')
                local NERIds = getNERIds();

                local nLastIdxMtWordVector = mtWordVector:size()[1]
                local sizeAppendDict = 50 - nLastIdxMtWordVector
                mtWordVector = appendRandomDataWeightMatrix(mtWordVector,sizeAppendDict,1)


                inputs, targets, features = loadDataSet("datasetT.txt",pairWordIds,NERIds, nLastIdxMtWordVector, sizeAppendDict)


                for i =1, #inputs do
                        print(inputs[i])
                        local xx = generateSentenceInputMatrix(inputs[i],50)

                        print(xx[i])
                end

        else

                -- lay ds NER
                local NERIds, NERLabels = getNERIds();

                -- Doc ma tran trong so WordtoVect(W2V)
                local pairWordIds, mtWordVector, pairIdWords = loadMatrixWordToVect(sDictName)

                -- Bo sung ma tran trong so W2V
                local nLastIdxMtWordVector = mtWordVector:size()[1]
                local sizeAppendDict = rawDataInputSize - nLastIdxMtWordVector
                mtWordVector = appendRandomDataWeightMatrix(mtWordVector,sizeAppendDict,1)

                if(g_isUseMaskZeroPadding == true)then
                        local mtPadding = initRandomDataWeightMatrix(1,(#mtWordVector)[2])
                        mtWordVector = torch.cat(mtPadding, mtWordVector, 1)
                end

                mtWeightInit = mtWordVector
                mtWordIds = pairWordIds
                mtIdWords = pairIdWords


                -- Doc bo du lieu dataset
                -- read dataset
                inputs, targets, features = loadDataSet(sDataSetName,pairWordIds,NERIds, nLastIdxMtWordVector, sizeAppendDict)
                if(g_isUseFeatureWord == true ) then 
                        if (g_iModelTest == 1) then 
                                g_nFeatureDims = #DataSet["featuresTrain"][1][1]
                        end 
                end

                -- tinh ti le bo du lieu train - test tren tung chu de
                -- calculate rate data train : test each label
                local _, rateTest = GetRateTrainingEachClass(DataSetGroup["targetsTest"])
                local _, rateTrain = GetRateTrainingEachClass(DataSetGroup["targetsTrain"])
                assert(#rateTest == g_nCountLabel and #rateTrain == g_nCountLabel,
                        "Loi: ma tran kq co so nhan tra ve khong hop le")

                for k, v in pairs(rateTrain) do
                        print(string.format("Ti le train/test nhan %s: %4.2f, %4.2f",
                                NERLabels[k], v/(rateTest[k]+v)*100.0, rateTest[k]/(rateTest[k]+v)*100.0))
                end

        end


end


