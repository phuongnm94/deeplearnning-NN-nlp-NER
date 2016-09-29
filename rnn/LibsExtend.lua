require 'nn'

---
-- string:plit(step) - chia xau string thanh mang phan cach boi cac ki tu 'step'
--
function string:split(sep)
        local sep, fields = sep or ":", {}
        local pattern = string.format("([^%s]+)", sep)
        self:gsub(pattern, function(c) fields[#fields+1] = c end)
        return fields
end

--
-- table:append(new table) mo rong bang tu 1 bang moi
--
function table:append(appendTable)      

        for _, v in pairs(appendTable) do 
                table.insert(self, v)
        end
        return self
end

function tableEx(tableOld)
    return setmetatable(tableOld, {__index = table})
end


--- ----------------------------------------------------------------------------
-- function check file Exist
-- -----------------------------------------------------------------------------
-- see if the file exists
-- http://lua-users.org/wiki/FileInputOutput

function isFileExists(sPathFile)
        local f = io.open(sPathFile, "rb")
        if f then f:close() end

        return f ~= nil

end

--- ----------------------------------------------------------------------------
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



--- ----------------------------------------------------------------------------
-- Kiem tra tu co ki tu so hay khong
--
-- @function [parent=#global] IsWordHasNumCharacter(sWord)
-- @param sWord tu can kiem tra
--
-- @return true - neu tu chua ki tu so
-- @return false - neu nguoc lai
function IsWordHasNumCharacter(sWord)

        local bRet = false
        local ret = string.find(sWord, "%d+")

        if ret ~= nil then
                bRet = true
                --                print(sWord)
        end

        return bRet
end

--- -----------------------------------------------------------------------------
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


--- ----------------------------------------------------------------------------
-- get Matrix lines from a file, returns an empty
--
-- @function [parent=#global] matrixExtractFrom(sPathFile)
-- @param sPathFile duong dan file du lieu chua ma tran
function matrixExtractFrom(sPathFile)

        local MatrixWordVectorRet = {}

        local elements
        local elementsDouble={}
        local WordIdRet = {}
        local IdWordRet = {}
        local sizeWordIds = 0

        if not isFileExists(sPathFile) then
                goto _EXIT_FUNCTION
        end

        for line in io.lines(sPathFile) do

                elements = line:split(" ")
                if(#elements == 0) then
                        print (line)
                        goto CONTINUE
                end

                sizeWordIds = sizeWordIds + 1
                WordIdRet[elements[1]:lower()] = sizeWordIds
                IdWordRet[sizeWordIds] = elements[1]:lower() 

                elements[1] = "0.0";
                elementsDouble = torch.DoubleStorage(elements)

                MatrixWordVectorRet[sizeWordIds] =
                        torch.totable(torch.DoubleStorage(elementsDouble, 2))

                ::CONTINUE::
        end

        ::_EXIT_FUNCTION::
        return WordIdRet, torch.Tensor(MatrixWordVectorRet), IdWordRet

end


--- ----------------------------------------------------------------------------
-- get Matrix lines from a file, returns an empty
--
--
-- @function [parent=#global] getDataSentenceFrom(sPathFileDataSet, pairWordIds, NERIds)
--
-- @param sPathFileDataSet Duong dan file DataSet
-- @param pairWordIds bang chua cac tu va id cua cac tu do
-- @param NERIds nhan phan loai cac loai NER
-- @param sizeAppendDict kich thuoc toi da dien vao cho cac tu Miss trong bo du lieu
--
-- @return inputs: Cac cau : {... {1, 5, 9, 8, 22,25}, {2,7,8,1,9}, ...}
-- @return targets: NER tuong ung : {... {1, 1, 1, 1, 1, 2}, {3,4,1,1,1}, ...}
function getDataSentenceFrom(sPathFileDataSet, pairWordIds, NERIds, nLastIdxWordIds, sizeAppendDict)

        local elements

        local tmpSentence, tmpNERSentence = {}, {}
        local idtmpWord
        local inputs, outputs = {}, {}
        local sizeSentencesInfo = {}
        local idxSentence, idxWord = 1 , 1
        local bFindNewSentence = true

        local lstNERIdsFound = {}


        if not isFileExists(sPathFileDataSet) then
                goto _EXIT_FUNCTION
        end

        for line in io.lines(sPathFileDataSet) do


                elements = line:split(" ")


                if #elements ~= 3 or elements[1] == "-DOCSTART-" then
                        -- print ('err -> ' .. line)
                        if(bFindNewSentence == false and idxWord > 1) then

                                -- them 1 cau vao bo du lieu
                                inputs[idxSentence] = tmpSentence
                                outputs[idxSentence] = tmpNERSentence

                                -- ghi lai thong tin ve kich thuoc so tu trong 1 cau
                                if(sizeSentencesInfo[idxWord] == nil) then
                                        sizeSentencesInfo[idxWord]={}
                                        sizeSentencesInfo[idxWord][1] = idxSentence
                                else
                                        sizeSentencesInfo[idxWord][#(sizeSentencesInfo[idxWord])+1] = idxSentence
                                end

                                -- reset data tmp sentence
                                tmpSentence = {}
                                tmpNERSentence = {}

                                -- tang con tro sang cau moi
                                idxSentence = idxSentence + 1
                                idxWord = 1
                        end

                        bFindNewSentence = true

                        goto CONTINUE
                end

                -- lay id cua tu trong tu dien
                idtmpWord = pairWordIds[elements[1]:lower()]
               
                if(idtmpWord ~= nil)then
                        -- lay tung gia tri id word
                        tmpSentence[idxWord] = idtmpWord
                else
                        -- them gia tri word id vao bang pairWordIds
                        if(sizeAppendDict ~= nil and sizeAppendDict > 0) then

                                sizeAppendDict = sizeAppendDict -1;
                                
                                nLastIdxWordIds = nLastIdxWordIds + 1
                                        
                                pairWordIds[elements[1]:lower()] = nLastIdxWordIds
                                tmpSentence[idxWord] = nLastIdxWordIds
                        else
                                print('miss -> ' .. elements[1])
                        end
                end
                tmpNERSentence[idxWord] = NERIds[elements[3]]


                -- thong ke cac nhan xuat hien trong tap dataset
                --                if(NERIds[elements[3]]==nil) then
                --                        print ('miss->['.. elements[3]..']')
                --                end
                --
                --                if(lstNERIdsFound ~= nil and lstNERIdsFound[elements[3]] == nil ) then
                --                        lstNERIdsFound[elements[3]] = 1
                --                else
                --                        lstNERIdsFound[elements[3]] = lstNERIdsFound[elements[3]] + 1
                --                end


                idxWord = idxWord +1;
                bFindNewSentence = false

                ::CONTINUE::
        end

        print('lstNERIdsFound : ', lstNERIdsFound)
        print ('Num Word Vector Append = '.. sizeAppendDict)

        ::_EXIT_FUNCTION::
        return inputs, outputs , sizeSentencesInfo

end



--- ----------------------------------------------------------------------------
-- get Matrix lines from a file, returns an empty
--
--
-- @function [parent=#global] getDataSentenceFrom2(PathFileDataSet, pairWordIds, NERIds, nLastIdxWordIds, sizeAppendDict)
--
-- @param sPathFileDataSet Duong dan file DataSet
-- @param pairWordIds bang chua cac tu va id cua cac tu do
-- @param NERIds nhan phan loai cac loai NER
-- @param sizeAppendDict kich thuoc toi da dien vao cho cac tu Miss trong bo du lieu
--
-- @return inputs: Cac cau : {... {1, 5, 9, 8, 22,25}, {2,7,8,1,9}, ...}
-- @return targets: NER tuong ung : {... {1, 1, 1, 1, 1, 2}, {3,4,1,1,1}, ...}
function getDataSentenceFrom2(sPathFileDataSet, pairWordIds, NERIds, nLastIdxWordIds, sizeAppendDict)

        local elements

        local tmpSentence, tmpNERSentence, tmpFeatureVector = {}, {}, {}
        local idtmpWord
        local inputs, outputs, features = {}, {}, {}
        local sizeSentencesInfo = {}
        local idxSentence, idxWord = 1 , 1
        local prevousWord = ""
   
        if not isFileExists(sPathFileDataSet) then
                goto _EXIT_FUNCTION
        end

        for line in io.lines(sPathFileDataSet) do

                local nLineWeight = #line

                -- Neu la 1 dong trong - them cau vao trong ds cau
                if nLineWeight == 0 or prevousWord == "." 
                then

                        -- Bo qua neu khong co tu nao trong cau dang xu ly
                        if #tmpSentence == 0 then goto FINISH_SAVE_SENTENCE end

                        -- them 1 cau vao bo du lieu
                        inputs[idxSentence] = tmpSentence
                        outputs[idxSentence] = tmpNERSentence
                        features[idxSentence] = tmpFeatureVector

                        -- ghi lai thong tin ve kich thuoc so tu trong 1 cau
                        if(sizeSentencesInfo[idxWord] == nil) then
                                sizeSentencesInfo[idxWord]={}
                                sizeSentencesInfo[idxWord][1] = idxSentence
                        else
                                sizeSentencesInfo[idxWord][#(sizeSentencesInfo[idxWord])+1] = idxSentence
                        end

                        -- reset data tmp sentence
                        tmpSentence, tmpNERSentence, tmpFeatureVector = {}, {}, {}

                        -- tang con tro sang cau moi
                        idxSentence = idxSentence + 1
                        idxWord = 1

                        
                        ::FINISH_SAVE_SENTENCE::
                        prevousWord = ""
                else
                        prevousWord = "" -- reset prevousWord       
                        elements = line:split(" ")
                        
                        if (    #elements < 3
                                or #(elements[1]) == 0
                                or #(elements[2]) == 0
                                or #(elements[3]) == 0
                                ) then  
                                --print(string.format("Error Data line = : %s", line))
                                goto FINISH_SAVE_WORD_TO_SENTENCE 
                        end
                        
                        if (elements[1]) == '-DOCSTART-' then goto FINISH_SAVE_WORD_TO_SENTENCE end
                        prevousWord = elements[1]:lower()
        
                        -- Them tu vao bo nho tam cua cau
                        idtmpWord = pairWordIds[prevousWord]
                        
                        if(idtmpWord ~= nil)then
                                -- lay tung gia tri id word
                                tmpSentence[idxWord] = idtmpWord
                        else
                                -- them gia tri word id vao bang pairWordIds
                                if(sizeAppendDict ~= nil and sizeAppendDict > 0) then

                                        sizeAppendDict = sizeAppendDict -1;
                                        pairWordIds[prevousWord] = nLastIdxWordIds + 1
                                        nLastIdxWordIds = nLastIdxWordIds + 1

                                        tmpSentence[idxWord] = nLastIdxWordIds

                                else
                                        print('miss -> ' .. elements[1])
                                end
                        end
                        
                        -- Them nhan cua tu vao tap nhan cua cau
                        tmpNERSentence[idxWord] = NERIds[elements[3]]
                        
                        -- Them vector features tung tu cho cau 
                        for i=3,1,-1 do
                                table.remove(elements, i)
                        end     
                        tmpFeatureVector[idxWord] = elements

                        idxWord = idxWord +1;
                        ::FINISH_SAVE_WORD_TO_SENTENCE::
                end
                
                -- thong ke cac nhan xuat hien trong tap dataset
                --                if(NERIds[elements[3]]==nil) then
                --                        print ('miss->['.. elements[3]..']')
                --                end
                --
                --                if(lstNERIdsFound ~= nil and lstNERIdsFound[elements[3]] == nil ) then
                --                        lstNERIdsFound[elements[3]] = 1
                --                else
                --                        lstNERIdsFound[elements[3]] = lstNERIdsFound[elements[3]] + 1
                --                end

        end

        print ('Num Word Vector Append = '.. sizeAppendDict)

        ::_EXIT_FUNCTION::
        return inputs, outputs , sizeSentencesInfo, features

end


--- ----------------------------------------------------------------------------
-- get Matrix lines from a file, returns an empty
--
--
-- @function [parent=#global] getDataSentenceFrom(sPathFileDataSet, pairWordIds, NERIds)
--
-- @param sPathFileDataSet Duong dan file DataSet
-- @param pairWordIds bang chua cac tu va id cua cac tu do
-- @param NERIds nhan phan loai cac loai NER
-- @param sizeAppendDict kich thuoc toi da dien vao cho cac tu Miss trong bo du lieu
--
-- @return inputs: Cac cau : {... {1, 5, 9, 8, 22,25}, {2,7,8,1,9}, ...}
-- @return targets: NER tuong ung : {... {1, 1, 1, 1, 1, 2}, {3,4,1,1,1}, ...}
function getFeatureSentenceFrom(sPathFileDataSet, pairWordIds)

        local elements

        local tmpSentence = {}
        local inputs = {}
        local idxSentence, idxWord = 1 , 1
        local bFindNewSentence = true



        if not isFileExists(sPathFileDataSet) then
                goto _EXIT_FUNCTION
        end

        for line in io.lines(sPathFileDataSet) do


                elements = line:split(" ")
                if #elements ~= 3 or elements[1] == "-DOCSTART-" then

                        if(bFindNewSentence == false and idxWord > 1) then

                                -- them 1 cau vao bo du lieu
                                inputs[idxSentence] = torch.Tensor(tmpSentence)

                                -- reset data tmp sentence
                                tmpSentence = {}

                                -- tang con tro sang cau moi
                                idxSentence = idxSentence + 1
                                idxWord = 1
                        end

                        bFindNewSentence = true
                        goto CONTINUE
                end

                if(pairWordIds[elements[1]] ~= nil) then
                        elements[1] = pairWordIds[elements[1]]
                else
                        elements[1] = -99999
                end
                
                table.remove(elements,1)
                tmpSentence[idxWord] = elements

                idxWord = idxWord +1;
                bFindNewSentence = false

                ::CONTINUE::
        end


        ::_EXIT_FUNCTION::
        return (inputs)

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

---
-- Unpack table to sub table 
-- 
function tableUnpackExtend(tblSrc, posStart, posEnd)
        
        assert(tblSrc ~= nil)
        posStart = math.max(1,posStart)
        posEnd = math.min(#tblSrc,posEnd)
        
        local posCurrent  = posStart
        local size = 6000
        --local tblRet = {}
        local tblRet = tableEx({})
        
        for posCurrent= posStart, posEnd, size do 
                tblRet:append({table.unpack(tblSrc,posCurrent,math.min(posCurrent + size -1, posEnd))})
        end
        
        --tblRet = nn.FlattenTable():forward(tblRet)
        return tblRet
end


--- ----------------------------------------------------------------------------
-- Buid table - {sizeSentence : {list Id sentence} } -
-- Xay dung cay chi muc nguoc kich thuoc => { idSentences }
--  
-- @function [parent=#global] groupSentenceSameSize(listSentence)
-- 
-- @param listSentence list sentence - ds cac cau
-- @return table indicates size to Id sentence
function groupSentenceSameSize(listSentence)

        local tblSizeToIdRet = {}
--        print (listSentence)
        for id, sentence in pairs(listSentence) do 
                if(tblSizeToIdRet[#sentence] == nil) then
                        tblSizeToIdRet[#sentence] = {}
                end 
                
                table.insert(tblSizeToIdRet[#sentence], id) 
        end
        
        return tblSizeToIdRet
end


