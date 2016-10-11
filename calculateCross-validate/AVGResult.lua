require ("ParamsParser")


---
-- string:plit(step) - chia xau string thanh mang phan cach boi cac ki tu 'step'
--
function string:split(sep)
        local sep, fields = sep or ":", {}
        local pattern = string.format("([^%s]+)", sep)
        self:gsub(pattern, function(c) fields[#fields+1] = c end)
        return fields
end


--- ----------------------------------------------------------------------------
-- function check file Exist
-- -----------------------------------------------------------------------------
-- see if the file exists
-- http://lua-users.org/wiki/FileInputOutput

local function isFileExists(sPathFile)
        local f = io.open(sPathFile, "rb")
        if f then f:close() end

        return f ~= nil

end

--- -----------------------------------------------------------------------------
-- get all lines from a file, returns an empty
-- list/table if the file does not exist
local function process_lines_from(file, functionCallback)

        local pairRets = {}
        local word = ""

        local elements

        if not isFileExists(file) then
                goto _EXIT_FUNCTION
        end

        for line in io.lines(file) do

                functionCallback(line)
        end

        ::_EXIT_FUNCTION::
        return pairRets

end


g_linesInfoResult = {}

local function functionCallback(line)

        if line == 'Khong ghep cap B- I- :  ' then
                g_linesInfoResult = {}
        end

        g_linesInfoResult[#g_linesInfoResult + 1] = line
end

local function readInfoTopic(lines, nCountTopic)

        local tblKoGhepCap, tblGhepCap = {0,0,0}, {0,0,0}

        for idx, val in pairs(lines) do
                if(idx >= 6 and idx <= 13) then

                        local idxType = 1
                        for number in string.gmatch(val,"(%d+)") do
                                
                                tblKoGhepCap[idxType] = tblKoGhepCap[idxType] + number
                                
                                idxType = idxType + 1
                                if(idxType == 4) then
                                        break
                                end
                        end


                end
                
                if(idx >= 20 and idx <= 27) then

                        local idxType = 1
                        for number in string.gmatch(val,"(%d+)") do
                                
                                tblGhepCap[idxType] = tblGhepCap[idxType] + number
                                
                                idxType = idxType + 1
                                if(idxType == 4) then
                                        break
                                end
                        end


                end
                
        end

        

        return {tblKoGhepCap, tblGhepCap}
end

function calAVGResult (sPatermNameLog, nCountTopic)

        local fileInfoRet={} 
        local sumInfo = {{0,0,0},{0,0,0}}
        
        for idxFile=1, 10 do

                local sFileLog = (string.format(sPatermNameLog, idxFile))
                print(sFileLog)
                
                process_lines_from(sFileLog,functionCallback)
                
--                print(g_linesInfoResult)        

                fileInfoRet[idxFile] = readInfoTopic(g_linesInfoResult,nCountTopic)
                
        end
        
        for k, info in pairs(fileInfoRet) do
                for idxType =1, 3 do 
                
                        sumInfo[1][idxType] = sumInfo[1][idxType] + info[1][idxType]
                        sumInfo[2][idxType] = sumInfo[2][idxType] + info[2][idxType]
                end                 
        
        end
        
        print(sumInfo)
        
        local p, r = sumInfo[1][1]*100.0/sumInfo[1][2], sumInfo[1][1]*100.0/sumInfo[1][3]
        print(string.format("Chua ghep cap: p = %6.2f, r = %6.2f, f1 = %6.2f",
                p, r, 2*p*r/(p+r)
                
                ))
        local p2, r2 = sumInfo[2][1]*100.0/sumInfo[2][2],sumInfo[2][1]*100.0/sumInfo[2][3]        
        print(string.format("Ghep cap: p = %6.2f, r = %6.2f, f1 = %6.2f",
                p2, r2, 2*p2*r2/(p2+r2)
                ))
end

opt = ParamsParser()
calAVGResult(opt.patermNameLog,9)
