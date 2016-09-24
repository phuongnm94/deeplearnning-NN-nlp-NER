require('rnn')
require('nn')
require('optim')
require('nninit')

require('ParamsParser')
require('InitData')
require('InitModel')
require('Trainning')
require('Testting')

-- ma tran trong so khoi tao (Word to vect)
mtWeightInit = nil

g_result = {}

-- data set
inputs, targets = {}, {}

-- pairNERIds
gNERIds = {}

-- define number word id
bUseWordIdPaddingForNumber = false
nWordIdPaddingForNumber = nil

function main()

        local opt = ParamsParser()
        local nRateTrainningSet = opt.trainRate

        local sNameNet = opt.nameNet
        

        -- Co bao chay batchInputs
        bIsRunInParalell = opt.isTrainBatchSentenceSameSize
        bIsUseOptimize = opt.isUseOptimizeGradient 

        -- hyper-parameters
        rawDataInputSize = 25000        -- so chieu vector
        rawFeatureInputSize = 126

        rho = 512                       -- so tu trong 1 cau ~ co the thay doi theo cau
        hiddenSize = 200                -- so chieu vector sinh boi word to vec
        nIndex = 9                      -- so nhan tu loai ~ O, B-ORG, B-PER, v...v....
        lr = opt.lr

        g_countLoopForOneBatch = opt.countLoopOneBatchSize
        g_batchSentenceSize = opt.batchSentenceSize
        g_countLoopAllData = opt.countLoopAllData
        g_iDataset = opt.iDataset
        g_isUseFeatureWord = opt.isUseFeatureWord
        g_trainRate = opt.trainRate
        
        print (opt)

        -- ---------------------------------------------------------------------------------------
        -- ---------------------------------------------------------------------------------------
        -- INIT DATA SET AND DICTIONARY
        -- ---------------------------------------------------------------------------------------
        -- ---------------------------------------------------------------------------------------
        if(g_isUseFeatureWord) then
                InitData('SubDict_vc.txt','features.txt')
        else
                InitData('SubDict_vc.txt','NonTag4type.tag')
        end
        print(#inputs)
        
        -- ---------------------------------------------------------------------------------------
        -- ---------------------------------------------------------------------------------------
        -- SETUP NERON NET
        -- ---------------------------------------------------------------------------------------
        -- ---------------------------------------------------------------------------------------
        netNN = InitModelNN(sNameNet,rawDataInputSize,hiddenSize,nIndex,mtWeightInit, rawFeatureInputSize)

        -- cai dat ham toi uu hoa gradient
        if(bIsUseOptimize) then 
                InitOptimizeConfig(netNN, opt)
        end


        -- ---------------------------------------------------------------------------------------
        -- ---------------------------------------------------------------------------------------
        -- SETUP LOSS FUNCTION -  ClassNLLCriterion
        -- ---------------------------------------------------------------------------------------
        -- ---------------------------------------------------------------------------------------
        criterion = nn.SequencerCriterion(nn.ClassNLLCriterion())


        -- ---------------------------------------------------------------------------------------
        -- ---------------------------------------------------------------------------------------
        -- TRAINING
        -- ---------------------------------------------------------------------------------------
        -- ---------------------------------------------------------------------------------------
        local nIndexStart, nIndexEnd  = 1,200
--        goto _BEGIN_TEST
        if bIsRunInParalell == false then
                nIndexStart, nIndexEnd = TrainningUseCrossvalidation(netNN,criterion,inputs,targets,nRateTrainningSet)
        else
                if(bIsUseOptimize) then
                        if(g_isUseFeatureWord == false) then
                                nIndexStart, nIndexEnd = TrainningUseOptimBatchCrossvalidation(netNN,criterion,inputsParallel,targetsParalel,nRateTrainningSet)
                        else
                                nIndexStart, nIndexEnd = TrainningUseOptimBatchFeaturesCrossvalidation(netNN,criterion,inputsParallel,targetsParalel,featuresParallel, nRateTrainningSet)
                        end
                else
                        nIndexStart, nIndexEnd = TrainningUseCrossvalidationParallel(netNN,criterion,inputsParallel,targetsParalel,nRateTrainningSet)
                end
        end



        -- ---------------------------------------------------------------------------------------
        -- ---------------------------------------------------------------------------------------
        -- TESTING
        -- ---------------------------------------------------------------------------------------
        -- ---------------------------------------------------------------------------------------
        ::_BEGIN_TEST::
        if bIsRunInParalell == false then
                TestUseCrossvalidation(netNN,inputs,targets,nIndex,nIndexStart,nIndexEnd)
        else
                TestUseCrossvalidationParallel(netNN,inputsParallel,targetsParalel,nIndex,nIndexStart,nIndexEnd, featuresParallel)
            
        end

        print('Tong hop ket qua TB: \n', g_result)
        ::_EXIT_FUNCTION_::

end
main()

