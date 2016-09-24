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
        cmd:option('--lr', 1e-2, 'learning rate - Toc do hoc cua mang ')
        cmd:option('--iDataset', 5, 'Cai dat train bo du lieu i')
        cmd:option('--trainRate', 0.9, 'train rate - Ti le tap hoc tren toan bo du lieu ')

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
