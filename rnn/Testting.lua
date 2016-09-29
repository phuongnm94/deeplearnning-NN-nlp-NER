function TestUseCrossvalidation(rnn, inputs, targets, nCountTopic, nIndexStart, nIndexEnd)

        --        local iteration = 10745
        local tblResultTrue = torch.LongTensor(nCountTopic):fill(0)
        local tblResultOut = torch.LongTensor(nCountTopic):fill(0)
        local tblNERDataset = torch.LongTensor(nCountTopic):fill(0)
        local sumTrue, sumOut, sumData = 0,0,0


        local tblResultTrue2 = torch.LongTensor(nCountTopic):fill(0)
        local tblResultOut2 = torch.LongTensor(nCountTopic):fill(0)
        local tblNERDataset2 = torch.LongTensor(nCountTopic):fill(0)
        local sumTrue2, sumOut2, sumData2 = 0,0,0

        local nSizeInputs = #inputs
        local iteration = nIndexEnd

        while iteration ~= nIndexStart do

                -- ----------------------------------------------------------------
                -- Khoi tao du lieu input cho 1 cau
                -- ----------------------------------------------------------------
                local wordCountInSentence = #inputs[iteration]


                local sentence = {}
                local sentenceNERDist = {}

                local outputs, err
                local gradOutputs, gradInputs
                local lstIdNEROut
                local idNEROut, idNERDist, _                      -- ket qua gan nhan cho tu



                sentence = torch.LongTensor({inputs[iteration]})
                sentenceNERDist = torch.LongTensor({targets[iteration]})

                outputs = rnn:forward(sentence)
                _, lstIdNEROut = outputs:topk(1, true)

                -- tong hop ket qua tung cau vao tap ket qua
                for i = 1, outputs:size()[2] do

                        idNERDist = sentenceNERDist[1][i]
                        idNEROut  = lstIdNEROut[1][i][1]
                        

                        if idNEROut == idNERDist then
                                tblResultTrue [idNERDist] = tblResultTrue[idNERDist]+1
                        end
                        tblResultOut[idNEROut] =  tblResultOut[idNEROut] + 1
                        tblNERDataset[idNERDist] = tblNERDataset[idNERDist] + 1

                end


                -- tong hop ket qua Theo nhan ngu nghia tung cau vao tap ket qua 2
                for i = 1, outputs:size()[2] do

                        idNERDist = sentenceNERDist[1][i]
                        idNEROut = lstIdNEROut[1][i][1]

                        if(idNEROut == 1 or idNEROut % 2 == 0) then
                                tblResultOut2[idNEROut] =  tblResultOut2[idNEROut] + 1
                        end

                        if(idNERDist == 1 or idNERDist % 2 == 0) then
                                tblNERDataset2[idNERDist] = tblNERDataset2[idNERDist] + 1
                        end

                        if(idNERDist == idNEROut) then
                                if(idNEROut == 1) then
                                        tblResultTrue2[idNEROut] =  tblResultTrue2[idNEROut] + 1
                                else
                                        if(idNEROut % 2 == 1) then goto _CONTINUE_FOR end

                                        if(i == outputs:size()[2] ) then
                                                tblResultTrue2[idNEROut] =  tblResultTrue2[idNEROut] + 1
                                                goto _CONTINUE_FOR
                                        end

                                        -- Kiem tra cac nhan I phia sau
                                        local bCheckAfterTagOK, bContinueCheck = false, true
                                        for j = i+1, outputs:size()[2]  do
                                                local _, idNERNextOut = outputs:topk(1, true)
                                                idNERNextOut = idNERNextOut[1][j][1]

                                                if(sentenceNERDist[1][j] ~= idNEROut + 1 ) then
                                                        tblResultTrue2[idNEROut] =  tblResultTrue2[idNEROut] + 1
                                                        break
                                                else
                                                        if(idNERNextOut ~= idNEROut + 1) then
                                                                break
                                                        end
                                                end

                                        end
                                end
                        end

                        ::_CONTINUE_FOR::
                end

                ::CONTINUE::
                iteration = iteration%nSizeInputs + 1
        end

        -- Ket qua tong hop chung tat ca cac nhan ngoai tru nhan O
        for i = 2, nCountTopic do
                sumData = sumData + tblNERDataset[i]
                sumData2 = sumData2+tblNERDataset2[i]

                sumOut = sumOut + tblResultOut[i]
                sumOut2 = sumOut2 + tblResultOut2[i]

                sumTrue = sumTrue + tblResultTrue[i]
                sumTrue2 = sumTrue2 + tblResultTrue2[i]
        end


        print '.....................................................'
        print 'So luong tu cua cac chu de'
        print '.....................................................'
        print(tblNERDataset)
        print(tblNERDataset2)

        print '.....................................................'
        print 'So luong tu gan nhan dung cua cac chu de'
        print '.....................................................'
        print(tblResultTrue)
        print(tblResultTrue2)

        print '.....................................................'
        print 'So luong tu duoc gan nhan cua cac chu de'
        print '.....................................................'
        print(tblResultOut)
        print(tblResultOut2)

        print '.....................................................'
        print 'Do phu'
        if(sumData ~= 0) then print(sumTrue*100.00/sumData) else print 'Ko co phan tu'end
        print '.....................................................'
        for i = 1, nCountTopic do
                if(tblNERDataset[i] ~= 0) then
                        print(tblResultTrue[i]*100.00/tblNERDataset[i])
                else
                        print 'Ko co phan tu'
                end
        end


        print '.....................................................'
        print 'Do phu - 2'
        if(sumData2 ~= 0) then print(sumTrue2*100.00/sumData2) else print 'Ko co phan tu'end
        print '.....................................................'
        for i = 1, nCountTopic do
                if(tblNERDataset2[i] ~= 0) then
                        print(tblResultTrue2[i]*100.00/tblNERDataset2[i])
                else
                        print 'Ko co phan tu'
                end
        end


        print '.....................................................'
        print 'Do chinh xac'
        if(sumOut ~= 0) then print(sumTrue*100.00/sumOut) else print 'Ko co phan tu'end
        print '.....................................................'
        for i = 1, nCountTopic do
                if(tblResultOut[i] ~= 0) then
                        print(tblResultTrue[i]*100.00/tblResultOut[i])
                else
                        print 'Ko co phan tu'
                end
        end

        print '.....................................................'
        print 'Do chinh xac - 2'
        if(sumOut2 ~= 0) then print(sumTrue2*100.00/sumOut2) else print 'Ko co phan tu'end
        print '.....................................................'
        for i = 1, nCountTopic do
                if(tblResultOut2[i] ~= 0) then
                        print(tblResultTrue2[i]*100.00/tblResultOut2[i])
                else
                        print 'Ko co phan tu'
                end
        end

        ::_EXIT_FUNCTION_::
        return {{sumData, sumData2}, {sumOut,sumOut2}, {sumTrue, sumTrue2}}
end

function TestUseCrossvalidationParallel2(rnn, inputs, targets, nCountTopic, nIndexStart, nIndexEnd, features)

        --        local iteration = 10745
        local tblResultTrue = torch.LongTensor(nCountTopic):fill(0)
        local tblResultOut = torch.LongTensor(nCountTopic):fill(0)
        local tblNERDataset = torch.LongTensor(nCountTopic):fill(0)
        local sumTrue, sumOut, sumData = 0,0,0


        local tblResultTrue2 = torch.LongTensor(nCountTopic):fill(0)
        local tblResultOut2 = torch.LongTensor(nCountTopic):fill(0)
        local tblNERDataset2 = torch.LongTensor(nCountTopic):fill(0)
        local sumTrue2, sumOut2, sumData2 = 0,0,0

        local nSizeInputs = #inputs
        local iteration = nIndexEnd

        while iteration ~= nIndexStart do

                -- ----------------------------------------------------------------
                -- Khoi tao du lieu input cho 1 cau
                -- ----------------------------------------------------------------
                local sentence, sentenceFeatures = {}, {}
                local sentenceNERDist = {}
                local sentenceOut = nil

                local outputs, err
                local gradOutputs, gradInputs
                local idNEROut, idNERDist, _                      -- ket qua gan nhan cho tu
                local nCountSentence = 1
                local nCountWordInSentence


                sentence = torch.Tensor(inputs[iteration]):t()
                
                -- tong hop ket qua tung cau vao tap ket qua
                nCountSentence = sentence:size()[1]
                nCountWordInSentence = sentence[1]:size()[1]
                
                if(g_isUseFeatureWord ~= nil and g_isUseFeatureWord == true) then
                        sentenceFeatures = torch.Tensor(features[iteration])
                        sentence = {sentence, sentenceFeatures}
                end
                
                sentenceNERDist = torch.Tensor(targets[iteration]):t()
                

                outputs = rnn:forward(sentence)
                _, sentenceOut = outputs:topk(1, true)
                

                for idxSentence =1 , nCountSentence do
                        for i = 1, nCountWordInSentence do

                                idNERDist = sentenceNERDist[idxSentence][i]
                                idNEROut  = sentenceOut[idxSentence][i][1]


                                if idNEROut == idNERDist then
                                        tblResultTrue [idNERDist] = tblResultTrue[idNERDist]+1
                                end
                                tblResultOut[idNEROut] =  tblResultOut[idNEROut] + 1
                                tblNERDataset[idNERDist] = tblNERDataset[idNERDist] + 1
                        end

                end


                -- tong hop ket qua Theo nhan ngu nghia tung cau vao tap ket qua 2
                for idxSentence = 1, nCountSentence do

                        for i = 1, nCountWordInSentence do

                                idNERDist = sentenceNERDist[idxSentence][i]
                                idNEROut  = sentenceOut[idxSentence][i][1]


                                if(idNEROut == 1 or idNEROut % 2 == 0) then
                                        tblResultOut2[idNEROut] =  tblResultOut2[idNEROut] + 1

                                end

                                if(idNERDist == 1 or idNERDist % 2 == 0) then
                                        tblNERDataset2[idNERDist] = tblNERDataset2[idNERDist] + 1

                                end

                                if(idNERDist == idNEROut) then
                                        if(idNEROut == 1) then
                                                tblResultTrue2[idNEROut] =  tblResultTrue2[idNEROut] + 1
                                        else
                                                -- Neu la nhan I-
                                                if(idNEROut % 2 == 1) then goto _CONTINUE_NEXT_WORD_ end

                                                -- Neu nhan B- la word cuoi cung trong cau
                                                if(i == nCountWordInSentence) then
                                                        tblResultTrue2[idNEROut] =  tblResultTrue2[idNEROut] + 1
                                                        goto _CONTINUE_NEXT_WORD_
                                                end

                                                -- Kiem tra cac nhan I- phia sau
                                                local j = 0
                                                for j = i+1, nCountWordInSentence do

                                                        local idNEROutNext  = sentenceOut[idxSentence][j][1]
                                                        local idNERDistNext = sentenceNERDist[idxSentence][j]

                                                        -- Neu cac tu phia sau trong Out va Dist deu khong phai la I- => +1
                                                        if(idNERDistNext ~= (idNEROut+1) and idNEROutNext ~= (idNEROut+1)) then
                                                                tblResultTrue2[idNEROut] =  tblResultTrue2[idNEROut] + 1
                                                                break
                                                        end

                                                        -- neu 1 trong 2 tu phia sau == IdNEROut + 1 nhung khac nhau => +0
                                                        if(idNERDistNext ~= idNEROutNext) then
                                                                break
                                                        end

                                                        -- Neu 2 tu (Out va Dist) = nhau va = idNEROut + 1 va la word cuoi cung
                                                        -- trong cau => +1
                                                        if(j == nCountWordInSentence) then
                                                                tblResultTrue2[idNEROut] =  tblResultTrue2[idNEROut] + 1
                                                        end

                                                end

                                        end
                                end
                                ::_CONTINUE_NEXT_WORD_::
                        end
                end

                ::CONTINUE::
                iteration = iteration%nSizeInputs + 1
        end

        -- Ket qua tong hop chung tat ca cac nhan ngoai tru nhan O
        for i = 2, nCountTopic do
                sumData = sumData + tblNERDataset[i]
                sumData2 = sumData2+tblNERDataset2[i]

                sumOut = sumOut + tblResultOut[i]
                sumOut2 = sumOut2 + tblResultOut2[i]

                sumTrue = sumTrue + tblResultTrue[i]
                sumTrue2 = sumTrue2 + tblResultTrue2[i]
        end

        print '................................................................'
        print 'Khong ghep cap B- I- : '
        print 'So tu gan nhan dung / So tu duoc gan nhan / So tu trong dataset '
        print (string.format('Precission = %6.2f, Recall = %6.2f ',
                        sumTrue*100.0/sumOut, sumTrue*100.0/sumData))
        print '................................................................'
        for i=1, nCountTopic do
                print(string.format("%6d, %6d, %6d,  P = %6.2f, R = %6.2f",
                        tblResultTrue[i], tblResultOut[i], tblNERDataset[i],
                        tblResultTrue[i]*100.00/tblResultOut[i], 
                        tblResultTrue[i]*100.00/tblNERDataset[i]
                        ))
        end


        print '................................................................'
        print 'Ghep cap BI- : '
        print 'So tu gan nhan dung / So tu duoc gan nhan / So tu trong dataset '
        print (string.format('Precission = %6.2f, Recall = %6.2f ',
                        sumTrue2*100.0/sumOut2, sumTrue2*100.0/sumData2))
        print '................................................................'
        for i=1, nCountTopic do
                print(string.format("%6d, %6d, %6d,  P = %6.2f, R = %6.2f",
                        tblResultTrue2[i], tblResultOut2[i], tblNERDataset2[i],
                        tblResultTrue2[i]*100.00/tblResultOut2[i], 
                        tblResultTrue2[i]*100.00/tblNERDataset2[i]
                        ))
        end

        print '................................................................'
        
        ::_EXIT_FUNCTION_::


        return {{sumData, sumData2}, {sumOut,sumOut2}, {sumTrue, sumTrue2}}
end

function TestUseCrossvalidationParallel(rnn, inputs, targets, nCountTopic, nIndexStart, nIndexEnd, features)

        --        local iteration = 10745
        local tblResultTrue = torch.LongTensor(nCountTopic):fill(0)
        local tblResultOut = torch.LongTensor(nCountTopic):fill(0)
        local tblNERDataset = torch.LongTensor(nCountTopic):fill(0)
        local sumTrue, sumOut, sumData = 0,0,0


        local tblResultTrue2 = torch.LongTensor(nCountTopic):fill(0)
        local tblResultOut2 = torch.LongTensor(nCountTopic):fill(0)
        local tblNERDataset2 = torch.LongTensor(nCountTopic):fill(0)
        local sumTrue2, sumOut2, sumData2 = 0,0,0

        local nSizeInputs = #inputs
        local iteration -- = nIndexEnd

        for iteration =1, nSizeInputs do --while iteration ~= nIndexStart do

                -- ----------------------------------------------------------------
                -- Khoi tao du lieu input cho 1 cau
                -- ----------------------------------------------------------------
                local sentence, sentenceFeatures = {}, {}
                local sentenceNERDist = {}
                local sentenceOut = nil

                local outputs, err
                local gradOutputs, gradInputs
                local idNEROut, idNERDist, _                      -- ket qua gan nhan cho tu
                local nCountSentence = 1
                local nCountWordInSentence


                sentence = torch.Tensor(inputs[iteration])--:t()
                
                -- tong hop ket qua tung cau vao tap ket qua
                nCountSentence = sentence:size()[1]
                nCountWordInSentence = sentence[1]:size()[1]
                
                if(g_isUseFeatureWord ~= nil and g_isUseFeatureWord == true) then
                        sentenceFeatures = torch.Tensor(features[iteration])
                        sentence = {sentence, sentenceFeatures}
                end
                
                sentenceNERDist = torch.Tensor(targets[iteration])--:t()
                

                outputs = rnn:forward(sentence)
                _, sentenceOut = outputs:topk(1, true)
                

                for idxSentence =1 , nCountSentence do
                        for i = 1, nCountWordInSentence do

                                idNERDist = sentenceNERDist[idxSentence][i]
                                idNEROut  = sentenceOut[idxSentence][i][1]
                                
                                -- bo qua neu la padding word 
                                if idNERDist == 0 then break end 
                                

                                if idNEROut == idNERDist then
                                        tblResultTrue [idNERDist] = tblResultTrue[idNERDist]+1
                                end
                                tblResultOut[idNEROut] =  tblResultOut[idNEROut] + 1
                                tblNERDataset[idNERDist] = tblNERDataset[idNERDist] + 1
                        end

                end


                -- tong hop ket qua Theo nhan ngu nghia tung cau vao tap ket qua 2
                for idxSentence = 1, nCountSentence do

                        for i = 1, nCountWordInSentence do

                                idNERDist = sentenceNERDist[idxSentence][i]
                                idNEROut  = sentenceOut[idxSentence][i][1]
                                
                                -- bo qua neu la padding word 
                                if idNERDist == 0 then break end 

                                if(idNEROut == 1 or idNEROut % 2 == 0) then
                                        tblResultOut2[idNEROut] =  tblResultOut2[idNEROut] + 1

                                end

                                if(idNERDist == 1 or idNERDist % 2 == 0) then
                                        tblNERDataset2[idNERDist] = tblNERDataset2[idNERDist] + 1

                                end

                                if(idNERDist == idNEROut) then
                                        if(idNEROut == 1) then
                                                tblResultTrue2[idNEROut] =  tblResultTrue2[idNEROut] + 1
                                        else
                                                -- Neu la nhan I-
                                                if(idNEROut % 2 == 1) then goto _CONTINUE_NEXT_WORD_ end

                                                -- Neu nhan B- la word cuoi cung trong cau
                                                if(i == nCountWordInSentence) then
                                                        tblResultTrue2[idNEROut] =  tblResultTrue2[idNEROut] + 1
                                                        goto _CONTINUE_NEXT_WORD_
                                                end

                                                -- Kiem tra cac nhan I- phia sau
                                                local j = 0
                                                for j = i+1, nCountWordInSentence do

                                                        local idNEROutNext  = sentenceOut[idxSentence][j][1]
                                                        local idNERDistNext = sentenceNERDist[idxSentence][j]

                                                        -- Neu cac tu phia sau trong Out va Dist deu khong phai la I- => +1
                                                        if(idNERDistNext ~= (idNEROut+1) and idNEROutNext ~= (idNEROut+1)) then
                                                                tblResultTrue2[idNEROut] =  tblResultTrue2[idNEROut] + 1
                                                                break
                                                        end

                                                        -- neu 1 trong 2 tu phia sau == IdNEROut + 1 nhung khac nhau => +0
                                                        if(idNERDistNext ~= idNEROutNext) then
                                                                break
                                                        end

                                                        -- Neu 2 tu (Out va Dist) = nhau va = idNEROut + 1 va la word cuoi cung
                                                        -- trong cau => +1
                                                        if(j == nCountWordInSentence) then
                                                                tblResultTrue2[idNEROut] =  tblResultTrue2[idNEROut] + 1
                                                        end

                                                end

                                        end
                                end
                                ::_CONTINUE_NEXT_WORD_::
                        end
                end

                ::CONTINUE::
                iteration = iteration%nSizeInputs + 1
        end

        -- Ket qua tong hop chung tat ca cac nhan ngoai tru nhan O
        for i = 2, nCountTopic do
                sumData = sumData + tblNERDataset[i]
                sumData2 = sumData2+tblNERDataset2[i]

                sumOut = sumOut + tblResultOut[i]
                sumOut2 = sumOut2 + tblResultOut2[i]

                sumTrue = sumTrue + tblResultTrue[i]
                sumTrue2 = sumTrue2 + tblResultTrue2[i]
        end

        print '................................................................'
        print 'Khong ghep cap B- I- : '
        print 'So tu gan nhan dung / So tu duoc gan nhan / So tu trong dataset '
        local P, R = sumTrue*100.0/sumOut, sumTrue*100.0/sumData
        print (string.format('Precission = %6.2f, Recall = %6.2f, F1 = %6.2f',
                        P, R, 2*P*R*1.00/(P+R)))
        print '................................................................'
        for i=1, nCountTopic do
                local Pi = tblResultTrue[i]*100.00/tblResultOut[i]
                local Ri = tblResultTrue[i]*100.00/tblNERDataset[i]
                print(string.format('%6d, %6d, %6d,  P = %6.2f, R = %6.2f, F1 = %6.2f',
                        tblResultTrue[i], tblResultOut[i], tblNERDataset[i],
                        Pi, 
                        Ri,
                        2*Pi*Ri*1.00/(Pi+Ri)
                        ))
        end


        print '................................................................'
        print 'Ghep cap BI- : '
        print 'So tu gan nhan dung / So tu duoc gan nhan / So tu trong dataset '
        local P2, R2 = sumTrue2*100.0/sumOut2, sumTrue2*100.0/sumData2
        print (string.format('Precission = %6.2f, Recall = %6.2f, F1 = %6.2f',
                        P2, R2, 2*P2*R2*1.00/(P2+R2)))
        print '................................................................'
        for i=1, nCountTopic do
                local Pi = tblResultTrue2[i]*100.00/tblResultOut2[i]
                local Ri = tblResultTrue2[i]*100.00/tblNERDataset2[i]
                print(string.format("%6d, %6d, %6d,  P = %6.2f, R = %6.2f, F1 = %6.2f",
                        tblResultTrue2[i], tblResultOut2[i], tblNERDataset2[i],
                        Pi,
                        Ri,        
                        2*Pi*Ri*1.00/(Pi+Ri)
                        ))
        end

        print '................................................................'
        
        ::_EXIT_FUNCTION_::


        return {{sumData, sumData2}, {sumOut,sumOut2}, {sumTrue, sumTrue2}}
end




function testing(rnn, inputs, targets, nCountTopic)

        local iteration = thresholdTraining + 1

        --        local iteration = 10745
        local tblResultTrue = torch.LongTensor(nCountTopic):fill(0)
        local tblResultOut = torch.LongTensor(nCountTopic):fill(0)
        local tblNERDataset = torch.LongTensor(nCountTopic):fill(0)
        local sumTrue, sumOut, sumData = 0,0,0


        local tblResultTrue2 = torch.LongTensor(nCountTopic):fill(0)
        local tblResultOut2 = torch.LongTensor(nCountTopic):fill(0)
        local tblNERDataset2 = torch.LongTensor(nCountTopic):fill(0)
        local sumTrue2, sumOut2, sumData2 = 0,0,0

        while iteration < #inputs do

                -- ----------------------------------------------------------------
                -- Khoi tao du lieu input cho 1 cau
                -- ----------------------------------------------------------------
                local wordCountInSentence = #inputs[iteration]


                local sentence = {}
                local sentenceNERDist = {}

                local outputs, err
                local gradOutputs, gradInputs
                local idNEROut, idNERDist, _                      -- ket qua gan nhan cho tu



                for i = 1, wordCountInSentence do
                        if(inputs[iteration][i] == nil) then
                                print('[Testing] Err 1 - iter' .. iteration)
                                goto CONTINUE
                        end
                        if(targets[iteration][i] == nil) then
                                print('[Testing] Err 2 - iter' .. iteration)
                                --                                targets[iteration][i] = 1
                                goto CONTINUE
                        end


                        sentence[i] = torch.LongTensor(1):fill(inputs[iteration][i])

                        sentenceNERDist[i] = torch.LongTensor(1):fill(targets[iteration][i])
                end

                outputs = rnn:forward(sentence)

                -- tong hop ket qua tung cau vao tap ket qua
                for i = 1, #outputs do

                        idNERDist = sentenceNERDist[i][1]
                        _, idNEROut  = outputs[i]:topk(1, true)

                        --                        print(outputs[i])
                        --
                        --                        print(idNERDist, idNEROut[1][1])

                        if idNEROut[1][1] == idNERDist then
                                tblResultTrue [idNERDist] = tblResultTrue[idNERDist]+1
                        end
                        tblResultOut[idNEROut[1][1]] =  tblResultOut[idNEROut[1][1]] + 1
                        tblNERDataset[idNERDist] = tblNERDataset[idNERDist] + 1

                end


                -- tong hop ket qua Theo nhan ngu nghia tung cau vao tap ket qua 2
                for i = 1, #outputs do

                        idNERDist = sentenceNERDist[i][1]
                        _, idNEROut  = outputs[i]:topk(1, true)
                        idNEROut = idNEROut[1][1]

                        if(idNEROut == 1 or idNEROut % 2 == 0) then
                                tblResultOut2[idNEROut] =  tblResultOut2[idNEROut] + 1
                        end

                        if(idNERDist == 1 or idNERDist % 2 == 0) then
                                tblNERDataset2[idNERDist] = tblNERDataset2[idNERDist] + 1
                        end

                        if(idNERDist == idNEROut) then
                                if(idNEROut == 1) then
                                        tblResultTrue2[idNEROut] =  tblResultTrue2[idNEROut] + 1
                                else
                                        if(idNEROut % 2 == 1) then goto _CONTINUE_FOR end

                                        if(i == #outputs) then
                                                tblResultTrue2[idNEROut] =  tblResultTrue2[idNEROut] + 1
                                                goto _CONTINUE_FOR
                                        end

                                        -- Kiem tra cac nhan I phia sau
                                        local bCheckAfterTagOK, bContinueCheck = false, true
                                        for j = i+1, #outputs do
                                                local _, idNERNextOut = outputs[j]:topk(1, true)
                                                idNERNextOut = idNERNextOut[1][1]

                                                if(sentenceNERDist[j][1] ~= idNEROut + 1 ) then
                                                        tblResultTrue2[idNEROut] =  tblResultTrue2[idNEROut] + 1
                                                        break
                                                else
                                                        if(idNERNextOut ~= idNEROut + 1) then
                                                                break
                                                        end
                                                end

                                        end
                                end
                        end

                        ::_CONTINUE_FOR::
                end

                ::CONTINUE::
                iteration = iteration + 1
        end

        -- Ket qua tong hop chung tat ca cac nhan ngoai tru nhan O
        for i = 2, nCountTopic do
                sumData = sumData + tblNERDataset[i]
                sumData2 = sumData2+tblNERDataset2[i]

                sumOut = sumOut + tblResultOut[i]
                sumOut2 = sumOut2 + tblResultOut2[i]

                sumTrue = sumTrue + tblResultTrue[i]
                sumTrue2 = sumTrue2 + tblResultTrue2[i]
        end


        print '...................................................................'
        print 'So luong tu gan nhan dung / So luong tu / So luong tu duoc gan nhan'
        print '...................................................................'
        for i=1, #tblNERDataset do
                print(string.format("%6d %6d %6d | %6d %6d %6d",
                        tblResultTrue[i], tblNERDataset[i], tblResultOut[i],
                        tblResultTrue2[i], tblNERDataset2[i], tblResultOut2[i]))
        end

        --
        --        print(tblNERDataset)
        --        print(tblNERDataset2)
        --
        --        print '.....................................................'
        --        print 'So luong tu gan nhan dung cua cac chu de'
        --        print '.....................................................'
        --        print(tblResultTrue)
        --        print(tblResultTrue2)
        --
        --        print '.....................................................'
        --        print 'So luong tu duoc gan nhan cua cac chu de'
        --        print '.....................................................'
        --        print(tblResultOut)
        --        print(tblResultOut2)

        print '.....................................................'
        print 'Do phu'
        if(sumData ~= 0) then print(sumTrue*100.00/sumData) else print 'Ko co phan tu'end
        print '.....................................................'
        for i = 1, nCountTopic do
                if(tblNERDataset[i] ~= 0) then
                        print(tblResultTrue[i]*100.00/tblNERDataset[i])
                else
                        print 'Ko co phan tu'
                end
        end


        print '.....................................................'
        print 'Do phu - 2'
        if(sumData2 ~= 0) then print(sumTrue2*100.00/sumData2) else print 'Ko co phan tu'end
        print '.....................................................'
        for i = 1, nCountTopic do
                if(tblNERDataset2[i] ~= 0) then
                        print(tblResultTrue2[i]*100.00/tblNERDataset2[i])
                else
                        print 'Ko co phan tu'
                end
        end


        print '.....................................................'
        print 'Do chinh xac'
        if(sumOut ~= 0) then print(sumTrue*100.00/sumOut) else print 'Ko co phan tu'end
        print '.....................................................'
        for i = 1, nCountTopic do
                if(tblResultOut[i] ~= 0) then
                        print(tblResultTrue[i]*100.00/tblResultOut[i])
                else
                        print 'Ko co phan tu'
                end
        end

        print '.....................................................'
        print 'Do chinh xac - 2'
        if(sumOut2 ~= 0) then print(sumTrue2*100.00/sumOut2) else print 'Ko co phan tu'end
        print '.....................................................'
        for i = 1, nCountTopic do
                if(tblResultOut2[i] ~= 0) then
                        print(tblResultTrue2[i]*100.00/tblResultOut2[i])
                else
                        print 'Ko co phan tu'
                end
        end

        ::_EXIT_FUNCTION_::
end
