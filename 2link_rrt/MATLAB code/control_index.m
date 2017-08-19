function j = control_index(t)
% dirty implementation :(
if (0.0 <= t) && (t < 0.025)
    j = 1;
elseif (0.025 <= t) && (t < 0.05)
    j = 2;
elseif (0.05 <= t) && (t < 0.075)
    j = 3;
elseif (0.075 <= t) && (t < 0.1)
    j = 4;
elseif (0.1 <= t) && (t < 0.125)
    j = 5;
elseif (0.125 <= t) && (t < 0.15)
    j = 6;
elseif (0.15 <= t) && (t < 0.175)
    j = 7;
elseif (0.175 <= t) && (t < 0.2)
    j = 8;
elseif (0.2 <= t) && (t < 0.225)
    j = 9;
elseif (0.225 <= t) && (t < 0.25)
    j = 10;
elseif (0.25 <= t) && (t < 0.275)
    j = 11;
elseif (0.275 <= t) && (t < 0.3)
    j = 12;
elseif (0.3 <= t) && (t < 0.325)
    j = 13;
elseif (0.325 <= t) && (t < 0.35)
    j = 14;
elseif (0.35 <= t) && (t < 0.375)
    j = 15;
elseif (0.375 <= t) && (t < 0.4)
    j = 16;
elseif (0.4 <= t) && (t < 0.425)
    j = 17;
elseif (0.425 <= t) && (t < 0.45)
    j = 18;
elseif (0.45 <= t) && (t < 0.475)
    j = 19;
elseif (0.475 <= t) && (t <= 0.5)
    j = 20;
end
% if rem(t,0.025)==0
%     j = (t/0.025);
% else
%     j = floor(t/0.025)+1;
% end
end