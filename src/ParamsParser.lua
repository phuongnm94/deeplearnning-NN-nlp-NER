require 'paths'
  
function ParamsParser()

        --[[ command line arguments ]]--
        cmd = torch.CmdLine()
      
        cmd:text()
        cmd:text('Chuong trinh phan loai tu - NER - brnn')
        cmd:text('Options:')
        cmd:text()

        -- training
        cmd:text('Training options')
        cmd:option('--lr', 0.1, 'learning rate - Toc do hoc cua mang ')
        cmd:option('--trainRate', 0.9, 'train rate - Ti le tap hoc tren toan bo du lieu ')
        cmd:option('--isTrainBatchSentenceSameSize', false, 'Ghep cac cau co cung kich thuoc trong 1 batchInputs')
        cmd:option('--isUseOptimizeGradient', false, 'Su dung Optimize Gradient')
        cmd:option('--isUseFeatureWord', false, 'Su dung dac trung ngon ngu')
        cmd:option('--batchSentenceSize', 30, 'so cau trong 1 batchInputs')
        cmd:option('--countLoopOneBatchSize', 5, 'so lan lap lai 1 batchInputs cau')
        cmd:option('--countLoopAllData', 100, 'so lan lap lai toan bo du lieu')
        cmd:option('--iDataset', 5, 'Cai dat train bo du lieu i')
        cmd:option('--nameNet', "brnnLstm", 'Cai dat ten mang Neron: rnn/rnnLstm/brnnLstm')


        -- loging
        cmd:text()
        cmd:text('Loging options')
        cmd:option('--isSaveLog', false, 'Ghi log ra file')
        cmd:option('--nameLog', "log", 'Ten file log')
        cmd:option('--hasTimeLog', false, 'xuat thoi gian trong log')

        cmd:text()


        local opt = cmd:parse(arg or {})

        -- create folder save log
        if(opt.isSaveLog == true) then
                
                if (opt.hasTimeLog == true) then
                        cmd:addTime('NER-brnn')
                end
                
                opt.rundir = cmd:string('logsFolder', {}, {dir=true})
                paths.mkdir(opt.rundir)

                cmd:log(opt.rundir .. '/'.. opt.nameLog, opt)

                
        end


        return opt
end
--ParamsParser()
