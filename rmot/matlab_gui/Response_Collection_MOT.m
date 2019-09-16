function [response_number]=Response_Collection_MOT(Available_Response_Options, Set_Size,frame_locations, stim_size, window, Choose_response_numbers)

%Response Collection
resp=0;
WaitforResponseLatency=GetSecs;  

'here'

response_accepted=0;

while resp==0 && response_accepted==0
    RestrictKeysForKbCheck([KbName('1!'),KbName('2@'),KbName('3#'),KbName('4$'),KbName('5%'),KbName('6^'),KbName('7&'),KbName('8*')]);
        keyIsDown = 0;
        code = 0;
        
        while(keyIsDown == 0)
            [keyIsDown, secs, keyCode] = KbCheck;
            keyIsDown=keyIsDown;
        end
        code = find(keyCode);
        
            if (code==KbName('1!'))
                response_number=1;
                resp=1;
            elseif(code==KbName('2@'))
                response_number=2;
                resp=1;
            elseif(code==KbName('3#'))
                response_number=3;
                resp=1;
            elseif(code==KbName('4$'))
                response_number=4;
                resp=1;
            elseif(code==KbName('5%'))
                response_number=5;
                resp=1;
            elseif(code==KbName('6^'))
                response_number=6;
                resp=1;
            elseif(code==KbName('7&'))
                response_number=7;
                resp=1;
            elseif(code==KbName('8*'))
                response_number=8;
                resp=1;
           
            end
        
response_number
is_this_option_available=ismember(response_number,Available_Response_Options);

if is_this_option_available==1
    response_accepted=1;
else
    response_accepted=0;
    resp=0;
end;

        
        
    end;
    
RestrictKeysForKbCheck([]); 
