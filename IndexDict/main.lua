require 'LibsExtend'
--requrie('math')

-- -----------------------------------------------------------------------------
-- function check file Exist
-- -----------------------------------------------------------------------------
-- see if the file exists
-- http://lua-users.org/wiki/FileInputOutput

function isFileExists(sPathFile)
        local f = io.open(sPathFile, "rb")
        if f then f:close() end

        return f ~= nil

end

-- -----------------------------------------------------------------------------
-- get all lines from a file, returns an empty
-- list/table if the file does not exist
function lines_from(file)

        local pairRets = {}

        if not isFileExists(file) then
                goto _EXIT_FUNCTION
        end

        for line in io.lines(file) do
                pairRets[#pairRets + 1] = line
        end

        ::_EXIT_FUNCTION::
        return pairRets

end


-- -----------------------------------------------------------------------------
-- get all lines from a file, returns an empty
-- list/table if the file does not exist
function lines_extract_from(file)

        local pairRets = {}
        local word = ""

        local elements

        if not isFileExists(file) then
                goto _EXIT_FUNCTION
        end

        for line in io.lines(file) do

                elements = line:split(" ")

                if(#elements > 0) then
                        pairRets[elements[1]:lower()] = line
                end

        end

        ::_EXIT_FUNCTION::
        return pairRets

end

-- -----------------------------------------------------------------------------
-- Guarded writeto (Lua4)
--   [f]: file name
-- Note: In Lua5 use io.output().
function writeto(hFileObj, sLine)

        if hFileObj then
                hFileObj:write(sLine, "\n")
        else
                print("Write file err: line = " .. sLine)
                goto _EXIT_FUNCTION
        end

        ::_EXIT_FUNCTION::

end


-- -----------------------------------------------------------------------------
function extractDict()

        -- set path file
        local sPathFileDict =
                --  'test.txt'
                '/media/phuongnm/USB/Data-rnn/vector-new-skipgram'
        local sPathFileDataSet = '/home/phuongnm/ECProjects/Lua/deeplearning-NN-nlp/sumNonTag4type.tag'

        local sPathFileSubDict = 'sumSubDict_vc.txt'
        local sPathFileMiss = 'sumMiss_vc.txt'

        -- crete file
        local fSubDict,err = io.open(sPathFileSubDict,"a")
        if err then goto _EXIT_FUNCTION end
        
        local fMiss,err = io.open(sPathFileMiss,"a")
        if err then goto _EXIT_FUNCTION end


        -- load file
        local pairDict = lines_extract_from(sPathFileDict);
        local pairDataSet = lines_extract_from(sPathFileDataSet);

        local iter = 0
        for k, v in pairs(pairDataSet) do

                local wordInfo = pairDict[k]
                if(wordInfo~= nil) then
                        writeto(fSubDict,wordInfo)
                else
                        writeto(fMiss, v)
                end

                if(iter % 1000 == 0) then
                        print(iter)
                end
                iter = iter+1;
        end

        if fSubDict ~= nil then fSubDict:close (); fSubDict = nil end
        if fMiss ~= nil then fMiss:close () ; fMiss = nil end

        ::_EXIT_FUNCTION::


end

extractDict()
