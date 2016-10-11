
function ParamsParser()

        --[[ command line arguments ]]--
        cmd = torch.CmdLine()
      
        cmd:text()
        cmd:text('Chuong trinh doc log')
        cmd:text('Options:')
        cmd:text()

        -- training
        cmd:option('--patermNameLog', "log%d.log", 'cau truc ten file log')

        cmd:text()


        local opt = cmd:parse(arg or {})

        
        return opt
end
--ParamsParser()
