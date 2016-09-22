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
        local pairWordIds, mtWordVector

        -- Doc ma tran vector cua cac tu
        pairWordIds, mtWordVector = matrixExtractFrom(sPathFileDict)

        if(bUseWordIdPaddingForNumber ~= nil and bUseWordIdPaddingForNumber == true) then

                -- Bo sung Padding - dung cho cac tu chua ki tu chu so
                nWordIdPaddingForNumber = (mtWordVector:size())[1] + 1
                mtWordVector = appendRandomDataWeightMatrix(mtWordVector,1,1)
        end


        ::_EXIT_FUNCTION::
        return pairWordIds, mtWordVector
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


        NERIds["B-LOC"] = 2
        NERIds["I-LOC"] = 3

        NERIds["B-ORG"] = 4
        NERIds["I-ORG"] = 5


        NERIds["B-PRO"] = 6
        NERIds["B-Pro"] = 6
        NERIds["I-PRO"] = 7

        NERIds["B-PER"] = 8
        NERIds["I-PER"] = 9


        return NERIds
end

---
-- Cai dat bo du lieu random cho cac tu khong co trong tu dien
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

        -- load data file
        local inputs, targets, features
        DataSet = {}

        inputs, targets, g_sizeSentencesInfo, features=
                getDataSentenceFrom2(sPathFileDataSet,pairWordIds,NERIds,nLastIdxMtWordVector, sizeAppendDict)
        assert(#inputs == #targets, 'Input vs Target is not same size')

        local nCountInputs = #inputs
        local nCountTestSet = math.max(1,(1 - g_trainRate) * nCountInputs);
        local nIndexTestStart = math.ceil(((g_iDataset-1)*(1-g_trainRate)*nCountInputs))%nCountInputs + 1
        local nIndexTestEnd = (nCountTestSet + nIndexTestStart )%nCountInputs
       
        print(string.format('Dataset test [%d/%d] : Cau %d - Cau %d', g_iDataset, 10, nIndexTestStart, nIndexTestEnd))
        DataSet["inputsTest"] = tableUnpackExtend(inputs,nIndexTestStart,nIndexTestEnd)
        DataSet["inputsTrain"] = {tableUnpackExtend(inputs,1,nIndexTestStart-1), tableUnpackExtend(inputs,nIndexTestEnd+1,nCountInputs)}
        DataSet["inputsTrain"] = nn.FlattenTable():forward(DataSet["inputsTrain"])
        
        DataSet["targetsTest"] = tableUnpackExtend(targets,nIndexTestStart,nIndexTestEnd)
        DataSet["targetsTrain"] =  {tableUnpackExtend(targets,1,nIndexTestStart-1), tableUnpackExtend(targets,nIndexTestEnd+1,nCountInputs)}
        DataSet["targetsTrain"] = nn.FlattenTable():forward(DataSet["targetsTrain"])

        if (bIsRunInParalell == true) then
                inputsParallel,targetsParalel, featuresParallel =  
                        splitDataSet(inputs,targets, features, sizeSentencesInfo, g_batchSentenceSize)
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
-- @function [parent=#global]  InitData(sDictName, sDataSetName, sPathFileFeature)
-- @param sDictName ten file tu dien
-- @param sDataSetName ten file dataset chua cac cau da duoc gan nhan cho tap hoc
-- @param sPathFileFeature ten file feature chua cac dac trung ngon ngu
function InitData(sDictName, sDataSetName)

        local bTestDebug = false
        if(bTestDebug == true) then

                pairWordIds, mtWordVector = loadMatrixWordToVect('te.txt')
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
                local NERIds = getNERIds();

                -- Doc ma tran trong so WordtoVect(W2V)
                local pairWordIds, mtWordVector = loadMatrixWordToVect(sDictName)

                -- Bo sung ma tran trong so W2V
                local nLastIdxMtWordVector = mtWordVector:size()[1]
                local sizeAppendDict = rawDataInputSize - nLastIdxMtWordVector
                mtWordVector = appendRandomDataWeightMatrix(mtWordVector,sizeAppendDict,1)

                mtWeightInit = mtWordVector
                mtWordIds = pairWordIds

                -- Doc bo du lieu dataset
                inputs, targets, features = loadDataSet(sDataSetName,pairWordIds,NERIds, nLastIdxMtWordVector, sizeAppendDict)

               
        end


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

                nCountElement = math.ceil((#dataset*0.9) + 1)
        end

        if(nIndexStart == nil ) then

                nIndexStart = 1
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

        for k, v in pairs(rateClassRet) do
                sumAllWord = sumAllWord + v
                nCountClass = nCountClass + 1
        end
        print(rateClassRet)
        mtRateRet = torch.DoubleTensor(nCountClass):fill(0.00)
        for idx = 1, nCountClass do
                mtRateRet[idx] = sumAllWord*1.00 / rateClassRet[idx]
        end
        return mtRateRet
end
