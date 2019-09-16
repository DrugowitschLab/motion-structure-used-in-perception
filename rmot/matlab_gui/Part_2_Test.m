%MOT with hierarchical structure
%Programmed by: Hrag Pailian
%Harvard University

%%%%%%%%%%%%%%%%%%%%%
%                   %
%        TEST       %
%                   %
%%%%%%%%%%%%%%%%%%%%%

clear all;
clc;
clear global;

%Randomize the random number generator
try
    rng('shuffle');
catch
    rand('twister',sum(100*clock));
end

Screen('Preference', 'SkipSyncTests', 1);

prompt = 'Input Speed-> 1.00, 1.25, 1.50, 1.75, 2.00, 2.25, 2.50, 2.75, 3.00: ';
speed=input(prompt,'s');

%Display speed at end of tracking
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                           %
%       PROMPT SCREEN       %
%                           %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% this is the prompt/input screen part
prompt = {'Subject number','Screen'};
def={'1','0'};
answer = inputdlg(prompt, 'Experimental setup information',1,def);
[subNum, Screen_no] = deal(answer{:});
Screen_no=str2num(Screen_no); %#ok<ST2NM>
subNum=str2num(subNum); %#ok<ST2NM>
ListenChar(1);
%HideCursor;
%%%%%%%%%

window=Screen('OpenWindow',Screen_no,[127 127 127]);
screenrect = Screen(window,'Rect'); % get the size of the display screen
[center_x,center_y]=RectCenter(screenrect); %find coordinates of the center of the screen

fontsize = 25; %fontsize
Screen('TextFont', window, 'Monaco');	% force fixed-Width font we want
Screen('TextSize', window, fontsize);		% force fontsize we want
Screen('TextStyle', window, 1);

HideCursor();

stim_size=40;
border_size=600;
border_color=[255 255 255];
black=[0 0 0];
green=[0 255 0];

cluster_green=[144 129 34];
cluster_pink=[246 37 111];
cluster_blue=[82 134 179];

cluster_colors={cluster_green cluster_pink cluster_blue};
target_thickness=6;
target_cue_size=12;

Block=1;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%FILENAME
response_filename = ['Response_File_Test_P' num2str(subNum) '.mat'];
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%
%                     %
%        SET UP       %
%                     %
%%%%%%%%%%%%%%%%%%%%%%%

main_path=pwd;

CenterText2(window,['Loading...',],[0 0 0],0,0);
Screen('Flip',window);
ResponseInfo=[];

CenterText2(window,['Ready. Press Enter.',],[0 0 0],0,0);
Screen('Flip',window);

WaitSecs(0.1);
KbWait;
WaitSecs(0.1);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%Predetermine trial order

counter=0;

for y=1:5
    
    %Latin Square Design for Counterbalancing 5 Conditions (determine order of trial blocks)
    if rem(subNum,5)==1
        Possible_Conditions={'counter','global','hierarchy_124','hierarchy_127','independent_test'};
    elseif rem(subNum,5)==2
        Possible_Conditions={'independent_test','counter','global','hierarchy_124','hierarchy_127'};
    elseif rem(subNum,5)==3
        Possible_Conditions={'hierarchy_127','independent_test','counter','global','hierarchy_124'};
    elseif rem(subNum,5)==4
        Possible_Conditions={'hierarchy_124','hierarchy_127','independent_test','counter','global'};
    elseif rem(subNum,5)==0
        Possible_Conditions={'global','hierarchy_124','hierarchy_127','independent_test','counter'};
    end;
    
    for x=1:30
        counter=counter+1;
        All_Trials{counter,1}=x;
        All_Trials{counter,2}=Possible_Conditions{y};
    end;
    
end;


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                               %
%        START EXPERIMENT       %
%                               %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

numTrials=size(All_Trials,1);

for i=1:numTrials
    
    %Predetermine trial order
    cd(main_path);
    cd('Trials')
    
    if strcmp(All_Trials{i,2},'counter')==1
        cd('counter')
        motion_hierarchy='counter';
    elseif strcmp(All_Trials{i,2},'global')==1
        cd('global')
        motion_hierarchy='global';
    elseif strcmp(All_Trials{i,2},'hierarchy_124')==1
        cd('hierarchy_124')
        motion_hierarchy='hierarchy_124';
    elseif strcmp(All_Trials{i,2},'hierarchy_127')==1
        cd('hierarchy_127')
        motion_hierarchy='hierarchy_127';
    elseif strcmp(All_Trials{i,2},'independent_test')==1
        cd('independent_test')
        motion_hierarchy='independent_test';
    end;
    
    
    
    
    %Load predetermined trial parameters
    cd(speed);
    chosen_trial=All_Trials{i,1};
    
    if chosen_trial<10
        trial_name=strcat('trial_0000',num2str(chosen_trial),'.mat');
    elseif chosen_trial<100
        trial_name=strcat('trial_000',num2str(chosen_trial),'.mat');
    elseif chosen_trial<1000
        trial_name=strcat('trial_00',num2str(chosen_trial),'.mat');
        
    end;
    
    load(trial_name);
    cd(main_path);
    
    textneedtobeshown1=strcat('Trial ',num2str(i));
    CenterText2(window,textneedtobeshown1,[100 100 100],0,-50);
    textneedtobeshown2=strcat(motion_hierarchy);
    CenterText2(window,textneedtobeshown2,[0 255 0],0,0);
    Screen('Flip',window);
    WaitSecs(1);
    Screen('Flip',window);
    
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    DrawFixationCross(window,center_x,center_y);
    Screen('Flip',window);
    WaitSecs(0.5); %
    
    cluster_colors=Shuffle(cluster_colors);
    
    temp_locations=X;
    x_offset=0;
    locations=temp_locations;
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %                               %
    %      SHOW INITIAL DISPLAY     %
    %                               %
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    for x=1:120 %For one second
        
        %Put up fixation cross and oval rink
        DrawFixationCross(window,center_x,center_y);
        
        centercoord = [center_x, center_y];
        left = centercoord(1) - (0.5 * 460.8);
        top = centercoord(2) - (0.5 * 460.8);
        right = centercoord(1) + (0.5 * 460.8);
        bottom = centercoord(2) + (0.5 * 460.8);
        coord = [left top right bottom];
        Screen('FrameOval', window, [0 0 0], coord, 2);
        
        %Put up items
        for p=1:size(locations,2)
            
            if strcmp(motion_hierarchy,'counter')==1
                
                if p==1 ||  p==2 || p==3
                    color=cluster_colors{1};
                elseif p==4
                    color=cluster_colors{2};
                elseif p==5 || p==6 || p==7
                    color=cluster_colors{3};
                end;
                
            else
                
                
                if p<=3
                    color=cluster_colors{1};
                elseif p>=3 && p<=6
                    color=cluster_colors{2};
                elseif p>=7
                    color=cluster_colors{3};
                end;
                
            end;
            
            centercoord = [locations(1,p,1)+x_offset,locations(1,p,2)];
            left = centercoord(1) - (0.5 * stim_size);
            top = centercoord(2) - (0.5 * stim_size);
            right = centercoord(1) + (0.5 * stim_size);
            bottom = centercoord(2) + (0.5 * stim_size);
            coord = [left top right bottom];
            
            Screen('filloval', window, color, coord);
            Screen('frameoval', window, [255 255 255], coord, 2);
            
            
        end;
        
        Screen('Flip',window);
        
    end;
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %                            %
    %      SHOW ITEMS MOVING     %
    %                            %
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    target_start_color_1=255;
    target_start_color_2=0;
    
    start_time=GetSecs;
    counter=0;
    
    color_setting=0;
    
    color_setting_2_counter=0;
    color_setting_3_counter=0;
    color_setting_4_counter=0;
    
    
    for x=1:(size(locations,1))
        
        %Settings for colors
        if x==1
            color_setting=1;
        elseif x==300
            color_setting=2;
            color_setting_2_counter=color_setting_2_counter+1;
            if color_setting_2_counter==1;
                ItemsOn_StartToMove_and_ClustersCued=GetSecs;
            end;
            
        elseif x==500
            color_setting=3;
            start_x=x;
            color_setting_3_counter=color_setting_3_counter+1;
            if color_setting_3_counter==1;
                ItemsOn_Moving_TargetCuesOn=GetSecs;
                
            end;
        end;
        
        if color_setting==3
            target_start_color_1=target_start_color_1-3;
            target_start_color_2=target_start_color_2+3;
        end;
        
        if target_start_color_1<=127
            target_start_color_1=127;
        end;
        
        if target_start_color_2>=127
            target_start_color_2=127;
        end;
        
        if target_start_color_1==127 && target_start_color_2==127
            color_setting=4;
            color_setting_4_counter=color_setting_4_counter+1;
            if color_setting_4_counter==1
                clc
                start_tracking_frames=x;
                start_tracking_time=GetSecs;
            end;
        end;
        
        %Put up fixation cross and oval rink
        DrawFixationCross(window,center_x,center_y);
        centercoord = [center_x, center_y];
        left = centercoord(1) - (0.5 * 460.8);
        top = centercoord(2) - (0.5 * 460.8);
        right = centercoord(1) + (0.5 * 460.8);
        bottom = centercoord(2) + (0.5 * 460.8);
        coord = [left top right bottom];
        Screen('FrameOval', window, [0 0 0], coord, 2);
        
        
        %DISPLAY MOVING ITEMS
        for p=1:size(locations,2)
            
            if color_setting==1 || color_setting==2
                
                if strcmp(motion_hierarchy,'counter')==1
                    
                    if p==1 ||  p==2 || p==3
                        color=cluster_colors{1};
                    elseif p==4
                        color=cluster_colors{2};
                    elseif p==5 || p==6 || p==7
                        color=cluster_colors{3};
                    end;
                    
                    
                else
                    
                    if p<=3
                        color=cluster_colors{1};
                    elseif p>=3 && p<=6
                        color=cluster_colors{2};
                    elseif p>=7
                        color=cluster_colors{3};
                    end;
                    
                    
                end;
                
                
            elseif color_setting==3
                color=black;
            elseif color_setting==4
                color=black;
            end;
            
            centercoord = [locations(x,p,1)+x_offset,locations(x,p,2)];
            left = centercoord(1) - (0.5 * stim_size);
            top = centercoord(2) - (0.5 * stim_size);
            right = centercoord(1) + (0.5 * stim_size);
            bottom = centercoord(2) + (0.5 * stim_size);
            coord = [left top right bottom];
            
            Screen('filloval', window, color, coord);
            Screen('frameoval', window, [255 255 255], coord, 2);
            
            cue_coord=[(left-target_cue_size) (top-target_cue_size) (right+target_cue_size) (bottom+target_cue_size)];
            
            
            if color_setting==2 || color_setting==3
                %INDICATE TARGETS
                if p==targets(1) || p==targets(2) || p==targets(3)
                    Screen('frameoval', window, [255 255 255], coord, 2);
                    Screen('framerect', window, [target_start_color_1 target_start_color_2 target_start_color_2], cue_coord, target_thickness);
                    
                end;
            end;
            
            
        end;
        
        
        Screen('Flip',window);
        
        end_tracking_time=GetSecs;
        
        if color_setting==4
            tracking_time=end_tracking_time-start_tracking_time;
        end;
        
        
        
    end;
    
    
    leave_loop_time=GetSecs;
    total_tracking_time=leave_loop_time-start_tracking_time;
    
    End_time=GetSecs;
    Elapsed=End_time-start_time;
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    final_locations=locations(x,:,:);
    frame_locations=final_locations;
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %                              %
    %      RESPONSE COLLECTION     %
    %                              %
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    Choose_response_numbers=randperm(7);
    % Choose_response_numbers=1:7;
    target_response_numbers=Choose_response_numbers(targets);
    Available_Response_Options=1:7;
    Set_Size=7;
    
    %Put up fixation cross and oval rink
    DrawFixationCross(window,center_x,center_y);
    
    centercoord = [center_x, center_y];
    left = centercoord(1) - (0.5 * 460.8);
    top = centercoord(2) - (0.5 * 460.8);
    right = centercoord(1) + (0.5 * 460.8);
    bottom = centercoord(2) + (0.5 * 460.8);
    coord = [left top right bottom];
    Screen('FrameOval', window, [0 0 0], coord, 2);
    
    response_option_colors={black black black black black black black black};
    for p=1:Set_Size
        color = response_option_colors{p};
        centercoord=[final_locations(1,p,1)+x_offset,final_locations(1,p,2)];
        left = centercoord(1) - (0.5 * stim_size);
        top = centercoord(2) - (0.5 * stim_size);
        right = centercoord(1) + (0.5 * stim_size);
        bottom = centercoord(2) + (0.5 * stim_size);
        coord = [left top right bottom];
        Screen('filloval', window, color, coord);
        Screen('frameoval', window, [255 255 255], coord, 2);
        Screen('DrawText', window, num2str(Choose_response_numbers(p)),centercoord(1)-8, centercoord(2)-16, [255 255 255]);
        
        
    end;
    
    
    
    Screen('Flip',window);
    
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    %First Response
    
    %Put up fixation cross and oval rink
    DrawFixationCross(window,center_x,center_y);
    
    centercoord = [center_x, center_y];
    left = centercoord(1) - (0.5 * 460.8);
    top = centercoord(2) - (0.5 * 460.8);
    right = centercoord(1) + (0.5 * 460.8);
    bottom = centercoord(2) + (0.5 * 460.8);
    coord = [left top right bottom];
    Screen('FrameOval', window, [0 0 0], coord, 2);
    
    
    start_response=GetSecs;
    [response_number]=Response_Collection_MOT(Available_Response_Options, Set_Size,frame_locations, stim_size, window, Choose_response_numbers);
    end_response=GetSecs;
    Response_1_Time=end_response-start_response;
    
    
    Response_1=response_number;
    Response_1_stim=find(Choose_response_numbers==Response_1);
    response_option_colors{Response_1_stim}=[0 0 255];
    Available_Response_Options(Available_Response_Options==Response_1)=[];
    
    for p=1:Set_Size
        color = response_option_colors{p};
        centercoord=[final_locations(1,p,1)+x_offset,final_locations(1,p,2)];
        left = centercoord(1) - (0.5 * stim_size);
        top = centercoord(2) - (0.5 * stim_size);
        right = centercoord(1) + (0.5 * stim_size);
        bottom = centercoord(2) + (0.5 * stim_size);
        coord = [left top right bottom];
        Screen('filloval', window, color, coord);
        Screen('frameoval', window, [255 255 255], coord, 2);
        
        Screen('DrawText', window, num2str(Choose_response_numbers(p)),centercoord(1)-8, centercoord(2)-16, [255 255 255]);
    end;
    
    
    Screen('Flip',window);
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    %Second Response
    
    %Put up fixation cross and oval rink
    DrawFixationCross(window,center_x,center_y);
    
    centercoord = [center_x, center_y];
    left = centercoord(1) - (0.5 * 460.8);
    top = centercoord(2) - (0.5 * 460.8);
    right = centercoord(1) + (0.5 * 460.8);
    bottom = centercoord(2) + (0.5 * 460.8);
    coord = [left top right bottom];
    Screen('FrameOval', window, [0 0 0], coord, 2);
    
    
    start_response=GetSecs;
    [response_number]=Response_Collection_MOT(Available_Response_Options, Set_Size,frame_locations, stim_size, window, Choose_response_numbers);
    end_response=GetSecs;
    Response_2_Time=end_response-start_response;
    
    Response_2=response_number;
    Response_2_stim=find(Choose_response_numbers==Response_2);
    response_option_colors{Response_2_stim}=[0 0 255];
    Available_Response_Options(Available_Response_Options==Response_2)=[];
    
    for p=1:Set_Size
        color = response_option_colors{p};
        centercoord=[final_locations(1,p,1)+x_offset,final_locations(1,p,2)];
        left = centercoord(1) - (0.5 * stim_size);
        top = centercoord(2) - (0.5 * stim_size);
        right = centercoord(1) + (0.5 * stim_size);
        bottom = centercoord(2) + (0.5 * stim_size);
        coord = [left top right bottom];
        Screen('filloval', window, color, coord);
        Screen('frameoval', window, [255 255 255], coord, 2);
        Screen('DrawText', window, num2str(Choose_response_numbers(p)),centercoord(1)-8, centercoord(2)-16, [255 255 255]);
    end;
    
    
    Screen('Flip',window);
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    %Third Response
    
    %Put up fixation cross and oval rink
    DrawFixationCross(window,center_x,center_y);
    
    centercoord = [center_x, center_y];
    left = centercoord(1) - (0.5 * 460.8);
    top = centercoord(2) - (0.5 * 460.8);
    right = centercoord(1) + (0.5 * 460.8);
    bottom = centercoord(2) + (0.5 * 460.8);
    coord = [left top right bottom];
    Screen('FrameOval', window, [0 0 0], coord, 2);
    
    
    start_response=GetSecs;
    [response_number]=Response_Collection_MOT(Available_Response_Options, Set_Size,frame_locations, stim_size, window, Choose_response_numbers);
    end_response=GetSecs;
    Response_3_Time=end_response-start_response;
    
    Response_3=response_number;
    Response_3_stim=find(Choose_response_numbers==Response_3);
    response_option_colors{Response_3_stim}=[0 0 255];
    Available_Response_Options(Available_Response_Options==Response_3)=[];
    
    for p=1:Set_Size
        color = response_option_colors{p};
        centercoord=[final_locations(1,p,1)+x_offset,final_locations(1,p,2)];
        left = centercoord(1) - (0.5 * stim_size);
        top = centercoord(2) - (0.5 * stim_size);
        right = centercoord(1) + (0.5 * stim_size);
        bottom = centercoord(2) + (0.5 * stim_size);
        coord = [left top right bottom];
        Screen('filloval', window, color, coord);
        Screen('frameoval', window, [255 255 255], coord, 2);
        Screen('DrawText', window, num2str(Choose_response_numbers(p)),centercoord(1)-8, centercoord(2)-16, [255 255 255]);
    end;
    
    
    Screen('Flip',window);
    
    % %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    %Check if they responded correctly or not
    
    response_1_correct=ismember(Response_1,target_response_numbers);
    response_2_correct=ismember(Response_2,target_response_numbers);
    response_3_correct=ismember(Response_3,target_response_numbers);
    
    clc
    
    Accuracy=((response_1_correct+ response_2_correct+ response_3_correct)/3)*100;
    
    %Put up fixation cross and oval rink
    DrawFixationCross(window,center_x,center_y);
    
    centercoord = [center_x, center_y];
    left = centercoord(1) - (0.5 * 460.8);
    top = centercoord(2) - (0.5 * 460.8);
    right = centercoord(1) + (0.5 * 460.8);
    bottom = centercoord(2) + (0.5 * 460.8);
    coord = [left top right bottom];
    Screen('FrameOval', window, [0 0 0], coord, 2);
    
    
    WaitSecs(0.5);
    
    for p=1:Set_Size
        color = response_option_colors{p};
        centercoord=[final_locations(1,p,1)+x_offset,final_locations(1,p,2)];
        left = centercoord(1) - (0.5 * stim_size);
        top = centercoord(2) - (0.5 * stim_size);
        right = centercoord(1) + (0.5 * stim_size);
        bottom = centercoord(2) + (0.5 * stim_size);
        coord = [left top right bottom];
        Screen('filloval', window, color, coord);
        Screen('frameoval', window, [255 255 255], coord, 2);
        Screen('DrawText', window, num2str(Choose_response_numbers(p)),centercoord(1)-8, centercoord(2)-16, [255 255 255]);
        
        
        box_left = centercoord(1) - (1 * stim_size);
        box_top = centercoord(2) - (1 * stim_size);
        box_right = centercoord(1) + (1 * stim_size);
        box_bottom = centercoord(2) + (1 * stim_size);
        box_coord = [box_left box_top box_right box_bottom];
        
        
        if p==targets(1) || p==targets(2) || p==targets(3)
            Screen('FrameRect', window, [255 0 0], box_coord, 3);
        end;
    end;
    
    
    
    Screen('Flip',window);
    WaitSecs(.5);
    
    Screen('Flip',window);
    
    WaitSecs(.5);
    Screen('Flip',window);
    
    if Accuracy==(0/3)*100
        NumItems_Correct=0;
    elseif Accuracy==(1/3)*100
        NumItems_Correct=1;
    elseif Accuracy==(2/3)*100
        NumItems_Correct=2;
    elseif Accuracy==(3/3)*100
        NumItems_Correct=3;
    end;
    
    total_Acc_PercentCorrect(i)=Accuracy;
    total_Acc_NumItemsCorrect(i)=NumItems_Correct;
    
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    %Save Response Info
    
    ResponseInfo(i).subNum=subNum;
    ResponseInfo(i).trial_number_actual=i;
    ResponseInfo(i).trial_number_name=chosen_trial;
    ResponseInfo(i).speed=speed;
    ResponseInfo(i).condition=motion_hierarchy;
    ResponseInfo(i).Accuracy_Percent_Correct=Accuracy;
    ResponseInfo(i).Accuracy_NumItems_Correct=NumItems_Correct;
    ResponseInfo(i).targets=targets;
    ResponseInfo(i).colors_cluster1_123=cluster_colors{1};
    ResponseInfo(i).colors_cluster2_456=cluster_colors{2};
    ResponseInfo(i).colors_cluster3_7=cluster_colors{3};
    ResponseInfo(i).tracking_duration=total_tracking_time;
    
    ResponseInfo(i).Choose_response_numbers=Choose_response_numbers;
    ResponseInfo(i).target_response_numbers=target_response_numbers;
    ResponseInfo(i).Response_1=Response_1;
    ResponseInfo(i).Response_1_stim=Response_1_stim;
    ResponseInfo(i).Response_2=Response_2;
    ResponseInfo(i).Response_2_stim=Response_2_stim;
    ResponseInfo(i).Response_3=Response_3;
    ResponseInfo(i).Response_3_stim=Response_3_stim;
    ResponseInfo(i).response_1_correct=response_1_correct;
    ResponseInfo(i).response_2_correct=response_2_correct;
    ResponseInfo(i).response_3_correct=response_3_correct;
    ResponseInfo(i).Response_1_Time=Response_1_Time;
    ResponseInfo(i).Response_2_Time=Response_2_Time;
    ResponseInfo(i).Response_3_Time=Response_3_Time;
    
    ResponseInfo(i).Event_1_ItemsOn_Stationary_and_ClustersCued=start_time;
    ResponseInfo(i).Event_2_ItemsOn_StartToMove_and_ClustersCued=ItemsOn_StartToMove_and_ClustersCued;
    ResponseInfo(i).Event_3_ItemsOn_Moving_TargetCuesOn=ItemsOn_Moving_TargetCuesOn;
    ResponseInfo(i).Event_4_TrackingPeriod_Starts_AllBlackDots=start_tracking_time;
    ResponseInfo(i).Event_5_TrackingPeriod_Ends_AllBlackDots=end_tracking_time;
    ResponseInfo(i).Start_Frame_FromOriginal=1;
    ResponseInfo(i).Start_Tracking_Frame=start_tracking_frames;
    ResponseInfo(i).End_Frame_FromOriginal=901;
    ResponseInfo(i).Block=Block;
    
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    if i==numTrials
        %Do nothing
        
    elseif rem(i,30)==0
        
        Screen('Flip',window);
        Block=Block+1;
        
        Trials_remaining=numTrials-i;
        textneedtobeshown=strcat('You have_',num2str(Trials_remaining),'_trials left');
        CenterText2(window,textneedtobeshown,[100 100 100],0,0);
        textneedtobeshown2=strcat('The Object Relationships/Motion Hierarchy Will Now Change!!!');
        CenterText2(window,textneedtobeshown2,[0 255 0],0,-50);
        
        textneedtobeshown=strcat('Please retrieve the experimenter.');
        CenterText2(window,textneedtobeshown,[0 0 0],0,-100);
        Screen('Flip',window);
        WaitSecs(60);
        KbWait;
        WaitSecs(0.1);
        
        
        %Update on subject tracker
        UpdateProgressTracker(webId, ...
            i/numTrials);
        
    elseif rem(i,10)==0
        
        Screen('Flip',window);
        textneedtobeshown=strcat('Take a quick break, if you feel that you need to.');
        CenterText2(window,textneedtobeshown,[0 0 0],0,0);
        textneedtobeshown2=strcat('Press the spacebar when you are ready to proceed.');
        CenterText2(window,textneedtobeshown2,[0 255 0],0,-50);
        Screen('Flip',window);
        WaitSecs(0.1);
        KbWait;
        WaitSecs(0.1);
        
    end
    
   
    
end;

save([response_filename],'ResponseInfo')

average_acc_PC=mean(total_Acc_PercentCorrect);
average_acc_NumItems=mean(total_Acc_NumItemsCorrect);

textneedtobeshown_1=strcat('Percent Correct:',num2str(average_acc_PC),'_Num Items Correct: ',num2str(average_acc_NumItems));
CenterText2(window,textneedtobeshown_1,[0 0 0],0,0);

textneedtobeshown_2=strcat('Speed:',(speed));
CenterText2(window,textneedtobeshown_2,[0 0 0],0,-40);

Screen('Flip',window);
WaitSecs(0.3);
KbWait;
WaitSecs(1);


ShowCursor;
Screen('CloseAll');
return;






%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

