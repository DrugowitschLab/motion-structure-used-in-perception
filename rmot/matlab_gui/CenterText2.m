function CenterText2 (win, str, color, xOffset, yOffset)

if nargin < 2
error('Usage: %s (win, str, [color], [xOffset], [yOffset])', mfilename);
end
if nargin < 3 || isempty(color)
color = BlackIndex(win);
end
if nargin < 4 || isempty(xOffset)
xOffset = 0;
end
if nargin < 5 || isempty(yOffset)
yOffset = 0;
end

rect = OffsetRect(CenterRect(Screen('TextBounds', win, str), Screen('Rect', win)), xOffset, yOffset);
Screen('DrawText', win, str, rect(RectLeft), rect(RectTop), color, [], 0);