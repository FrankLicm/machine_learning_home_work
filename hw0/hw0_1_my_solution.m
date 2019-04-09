syms a11 a12 a13 a21 a22 a23 a31 a32 a33
A = [a11 a12 a13; a21 a22 a23; a31 a32 a33];
v1 = [sqrt(3)/2;0.5;0];
v2 = [0;0.5;sqrt(3)/2];
v3 = [1/sqrt(2);0;-1/sqrt(2)];
v4 = [-1/sqrt(2);1/sqrt(2);0];
v6 = [v3, v4];
v7 = [v1, v2];
[s11,s12,s13,s21,s22,s23,s31,s32,s33,params,conditions] = solve([v6==A*v7,v7 == inv(A)*v6],[a11,a12,a13,a21,a22,a23,a31,a32,a33],'ReturnConditions',true);
A = [s11 s12 s13;s21 s22 s23; s31 s32 s33]
params
conditions
