R = 2.0;                % Ohms
L = 0.5;                % Henrys
Km = 0.015;               % torque constant
Kb = 0.015;               % back emf constant
Kf = 0.2;               % Nms
J = 0.02;               % kg.m^2/s^2

%%
% Solve for LQR controller

h1 = tf(Km,[L R]);            % armature
h2 = tf(1,[J Kf]);            % eqn of motion
dcm = ss(h2) * [h1 , 1];      % w = h2 * (h1*Va + Td)
dcm = feedback(dcm,Kb,1,1);   % close back emf loop
dc_aug = [1 ; tf(1,[1 0])] * dcm(1); % add output w/s to DC motor model
K_lqr = lqry(dc_aug,[1 0;0 20],0.1);
LQR_Cont = K_lqr * append(tf(1,[1 0]),1,1);   % compensator including 1/s

% A good controller 
Hinf_Cont = tf(84*[.233 1],[.0357 1 0]);
% Cont = tf(84*[.233],[.0357 1]);
PI_Cont = tf([4.10 5],[1 0]);

%% System with unmodeled uncertainty
R = ureal('R',2,'Percentage',40);
L = ureal('L',0.5,'Percentage',40);
K = ureal('K',0.015,'Range',[0.012 0.019]);
Km = K;
Kb = K;
Kf = ureal('Kf',0.2,'Percentage',50);

H = [1;0;Km] * tf(1,[L R]) * [1 -Kb] + [0 0;0 1;0 -Kf];
H.InputName = {'AppliedVoltage';'AngularSpeed'};
H.OutputName = {'Current';'AngularSpeed';'RotorTorque'};
J = 0.02*(1 + ultidyn('Jlti',[1 1],'Type','GainBounded','Bound',0.15,'SampleStateDim',4));
Pall = lft(H,tf(1,[1 0])/J);

%% Foward transfer function (open loop)
Hinf_OL_all = Pall*Hinf_Cont;         %mimo
Hinf_OL = Hinf_OL_all(2,:);           %siso
LQR_OL_all = Pall*LQR_Cont; %mimo
LQR_OL = LQR_OL_all(2,1);   %siso
PI_OL_all = Pall*PI_Cont;         %mimo
PI_OL = PI_OL_all(2,:);           %siso

figure
bode(Hinf_OL.NominalValue);
margin(Hinf_OL.NominalValue);
%title('Gain and phase margins')
figure
bode(LQR_OL.NominalValue)
margin(LQR_OL.NominalValue);
%title('LQR gain and phase margins')
figure
bode(PI_OL.NominalValue);
margin(PI_OL.NominalValue);
%title('PI Gain and phase margins')

Hinf_DM = diskmargin(Hinf_OL.NominalValue)
Hinf_wcDM = wcdiskmargin(Hinf_OL,'siso')
mag2db(Hinf_wcDM.GainMargin)
LQR_DM = diskmargin(LQR_OL.NominalValue)
LQR_wcDM = wcdiskmargin(LQR_OL,'siso')
mag2db(LQR_wcDM.GainMargin)
PI_DM = diskmargin(PI_OL.NominalValue)
PI_wcDM = wcdiskmargin(PI_OL,'siso')
mag2db(PI_wcDM.GainMargin)

%% Sensitiviy (closed loop)
Hinf_S = feedback(1,Hinf_OL);
figure
bodemag(Hinf_S,Hinf_S.Nominal)
legend('Hinf_Samples','Hinf_Nominal')
figure
step(Hinf_S,Hinf_S.Nominal)
title('Hinf Disturbance Rejection')
legend('Hinf_Samples','Hinf_Nominal')

LQR_S = feedback(1,LQR_OL);
figure
bodemag(LQR_S,LQR_S.Nominal)
legend('LQR_Samples','LQR_Nominal')
figure
step(LQR_S,LQR_S.Nominal)
title('LQR Disturbance Rejection')
legend('LQR_Samples','LQR_Nominal')

PI_S = feedback(1,PI_OL);
figure
bodemag(PI_S,PI_S.Nominal)
legend('PI_Samples','PI_Nominal')
figure
step(PI_S,PI_S.Nominal)
title('PI Disturbance Rejection')
legend('PI_Samples','PI_Nominal')

[Hinf_maxgain,Hinf_worstuncertainty] = wcgain(Hinf_S);
Hinf_maxgain
Hinf_Sworst = usubs(Hinf_S,Hinf_worstuncertainty);
norm(Hinf_Sworst,inf)
Hinf_maxgain.LowerBound

[lqr_maxgain,lqr_worstuncertainty] = wcgain(LQR_S);
lqr_maxgain
lqr_Sworst = usubs(LQR_S,lqr_worstuncertainty);
norm(lqr_Sworst,inf)
lqr_maxgain.LowerBound

[PI_maxgain,PI_worstuncertainty] = wcgain(PI_S);
PI_maxgain
PI_Sworst = usubs(PI_S,PI_worstuncertainty);
norm(PI_Sworst,inf)
PI_maxgain.LowerBound

figure
step(Hinf_Sworst,Hinf_S.NominalValue,6);hold all
step(lqr_Sworst,LQR_S.NominalValue,6);hold all
step(PI_Sworst,PI_S.NominalValue,6);
title('Disturbance Rejection')
legend('Hinf_Worst-case','Hinf_Nominal','LQR_Worst-case','LQR_Nominal','PI_Worst-case','PI_Nominal')

figure
bodemag(Hinf_Sworst,Hinf_S.NominalValue);hold all
bodemag(lqr_Sworst,LQR_S.NominalValue);hold all
bodemag(PI_Sworst,PI_S.NominalValue)
%title('Disturbance Rejection')
legend('Hinf_Worst-case','Hinf_Nominal','LQR_Worst-case','LQR_Nominal','PI_Worst-case','PI_Nominal')


figure

subplot(3,2,1); 
bodemag(Hinf_S,Hinf_S.Nominal)
legend('Hinf_Samples','Hinf_Nominal','Location', 'Best')

subplot(3,2,2); 
step(Hinf_S,Hinf_S.Nominal)
title('Hinf Disturbance Rejection')
legend('Hinf_Samples','Hinf_Nominal','Location', 'Best')

subplot(3,2,3); 
bodemag(LQR_S,LQR_S.Nominal)
legend('LQR_Samples','LQR_Nominal','Location', 'Best')

subplot(3,2,4); 
step(LQR_S,LQR_S.Nominal)
title('LQR Disturbance Rejection')
legend('LQR_Samples','LQR_Nominal','Location', 'Best')

subplot(3,2,5); 
bodemag(PI_S,PI_S.Nominal)
legend('PI_Samples','PI_Nominal','Location', 'Best')

subplot(3,2,6); 
step(PI_S,PI_S.Nominal)
title('PI Disturbance Rejection')
legend('PI_Samples','PI_Nominal','Location', 'Best')