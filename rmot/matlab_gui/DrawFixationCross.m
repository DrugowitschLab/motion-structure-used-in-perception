function DrawFixationCross(window,x,y)

% w is your window thing from psychtoolbox
% (x,y) is the where you want the center of the cross to be

%Half width of the line of the fixation cross
fixationpthalfwidth=8;
%To move fixation cross to the left, use center_x - value.  Higher value =
%more left
fixationcenterx=x;


Screen('DrawLine', window, [0 0 0], fixationcenterx+fixationpthalfwidth, y, fixationcenterx-fixationpthalfwidth, y, 3);
Screen('DrawLine', window, [0 0 0], fixationcenterx, y+fixationpthalfwidth, fixationcenterx, y-fixationpthalfwidth , 3);  