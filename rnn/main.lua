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


function main()

        local opt = ParamsParser()
        local nRateTrainningSet = opt.trainRate

        local sNameNet = opt.nameNet


        -- Co bao chay batchInputs
        bIsRunInParalell = opt.isTrainBatchSentenceSameSize
        bIsUseOptimize = opt.isUseOptimizeGradient

        -- hyper-parameters
        rawDataInputSize = 10000        -- so chieu vector 

        rho = 512                       -- so tu trong 1 cau ~ co the thay doi theo cau
        hiddenSize = 200                -- so chieu vector sinh boi word to vec
        g_nCountLabel = 9               -- so nhan tu loai ~ O, B-ORG, B-PER, v...v....
        lr = opt.lr

        g_countLoopForOneBatch = opt.countLoopOneBatchSize
        g_batchSentenceSize = opt.batchSentenceSize
        g_countLoopAllData = opt.countLoopAllData
        g_iDataset = opt.iDataset
        g_trainRate = opt.trainRate
        g_nCountLabel = g_nCountLabel
        g_isUseMaskZeroPadding = opt.isUseMaskZeroPadding
        g_isReparseBalanceData = opt.isReparseBalanceData
        g_isUseFeatureWord = opt.isUseFeatureWord
        g_isUseCuda = opt.cuda
        g_iModelTest = opt.iModelTest
        g_nFeatureDims = nil
        g_nFeatureSize = nil

        print (opt)

        -- ---------------------------------------------------------------------
        -- ---------------------------------------------------------------------
        -- INIT DATA SET AND DICTIONARY
        -- ---------------------------------------------------------------------
        -- ---------------------------------------------------------------------
        if(g_isUseFeatureWord) then
                InitData('SubDict_vc.txt','features.txt')
        else
                -- InitData('sumSubDict_vc.txt','sumNonTag4type.tag')
                InitData('SubDict_trial.txt','data_trial.tag')
        end



        -- ---------------------------------------------------------------------
        -- ---------------------------------------------------------------------
        -- SETUP NERON NET
        -- ---------------------------------------------------------------------
        -- ---------------------------------------------------------------------
        netNN = InitModelNN(sNameNet,rawDataInputSize,hiddenSize,g_nCountLabel,
                mtWeightInit, g_nFeatureDims)
        --print(netNN)
        
        -- cai dat ham toi uu hoa gradient
        if(bIsUseOptimize) then
                InitOptimizeConfig(netNN, opt)
        end



        -- ---------------------------------------------------------------------
        -- ---------------------------------------------------------------------
        -- SETUP LOSS FUNCTION -  ClassNLLCriterion
        -- ---------------------------------------------------------------------
        -- ---------------------------------------------------------------------
        -- Tinh ma tran trong so khi tap hoc thay doi
        local mtRateClassTraining = GetRateTrainingEachClass(
                DataSetGroup["targetsTrain"],   -- dataset
                1,                              -- index start
                #DataSetGroup["targetsTrain"]   -- size
        )
        criterion = nn.ClassNLLCriterion(mtRateClassTraining)
        if g_isUseMaskZeroPadding then
                criterion = nn.SequencerCriterion(nn.MaskZeroCriterion(criterion, 1))
        else
                criterion = nn.SequencerCriterion(criterion)
        end
        if (g_isUseCuda == true)then 
                require 'cutorch'
                require 'cunn'
                criterion = criterion:cuda()
        end

        -- ---------------------------------------------------------------------
        -- ---------------------------------------------------------------------
        -- TRAINING
        -- ---------------------------------------------------------------------
        -- ---------------------------------------------------------------------
        local nIndexStart, nIndexEnd  = 1,200
        --goto _BEGIN_TEST
        if(bIsUseOptimize) then
                if(g_isUseFeatureWord == false) then
                        nIndexStart, nIndexEnd = TrainningUseOptimBatchCrossvalidation(
                                netNN,
                                criterion,
                                DataSetGroup["inputsTrain"],
                                DataSetGroup["targetsTrain"],
                                nRateTrainningSet)
                else
                        nIndexStart, nIndexEnd = TrainningUseOptimBatchFeaturesCrossvalidation(
                                netNN,
                                criterion,
                                DataSetGroup["inputsTrain"],
                                DataSetGroup["targetsTrain"],
                                DataSetGroup["featuresTrain"],
                                nRateTrainningSet)

                end
        else
                nIndexStart, nIndexEnd = TrainningUseCrossvalidationParallel(
                        netNN,
                        criterion,
                        DataSetGroup["inputsTrain"],
                        DataSetGroup["targetsTrain"],
                        nRateTrainningSet)
        end

        

        -- ---------------------------------------------------------------------
        -- ---------------------------------------------------------------------
        -- TESTING
        -- ---------------------------------------------------------------------
        -- ---------------------------------------------------------------------
        ::_BEGIN_TEST::
        --netNN = torch.load('ner_data_vc.model10')
        
        TestUseCrossvalidationParallel(
                netNN,
                DataSetGroup["inputsTest"],
                DataSetGroup["targetsTest"],
                g_nCountLabel,
                nIndexStart,nIndexEnd,
                DataSetGroup["featuresTest"])

        
        print('Tong hop ket qua TB: \n', g_result)
        ::_EXIT_FUNCTION_::

end
main()

