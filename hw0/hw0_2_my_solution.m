%2.a
syms X Y
[a,b] = solve([2*X^2+4*Y*(X+Y^2-7)+2*Y-22 == 0,4*X*(X^2+Y-11)+2*X+2*Y^2-14 == 0],[X Y]);
f = (X*X+Y-11)^2 + (X+Y*Y-7)^2;
fc = fcontour(f);
fc.LevelStep = 5;
hold on;
axis([-5,5,-5,5])
saddle_x = [a(2);a(3);a(5);a(6)];
saddle_y = [b(2);b(3);b(5);b(6)];
plot(saddle_x,saddle_y,'*g');
minima_x = [a(1);a(7);a(8);a(9)];
minima_y = [b(1);b(7);b(8);b(9)];
plot(minima_x,minima_y,'*r');
maxima_x = a(4);
maxima_y = b(4);
plot(maxima_x,maxima_y,'*b');
legend("cost","saddle","minima","maxima");
df_dx = diff(f, X);
df_dy = diff(f, Y);

%2.b
for k=1:10
    x(1) = -5+10*rand();
    y(1) = 5+10*rand();
    e = 10^(-8); 
    i = 1;   
    J = [subs(df_dx,[X,Y], [x(1),y(1)]) subs(df_dy, [X,Y], [x(1),y(1)])]; 
    S = -(J);
    r = 0.001;
    while norm(J) > e && i < 100000 
        I = [x(i),y(i)]';
        x(i+1) = I(1)+r*S(1); 
        y(i+1) = I(2)+r*S(2); 
        i = i+1;
        J = [subs(df_dx,[X,Y], [x(i),y(i)]) subs(df_dy, [X,Y], [x(i),y(i)])];
        S = -(J); 
    end
    h(i) = plot(x,y,'.m');
    set(h(i),'handlevisibility','off');
    disp(k)
end

% 2.c
df_dx = diff(f, X);
df_dy = diff(f, Y);
ddf_ddx = diff(df_dx,X);
ddf_ddy = diff(df_dy,Y);
ddf_dxdy = diff(df_dx,Y);
ddf_dydx = diff(df_dy,X);
for k=1:10
    x(1) = -5+10*rand();
    y(1) = -5+10*rand();
    e = 10^(-8); 
    r = 1e-2;
    i = 1; 
    J = [subs(df_dx,[X,Y], [x(1),y(1)]) subs(df_dy, [X,Y], [x(1),y(1)])]; 
    ddf_ddx_1 = subs(ddf_ddx, [X,Y], [x(1),y(1)]);
    ddf_ddy_1 = subs(ddf_ddy, [X,Y], [x(1),y(1)]);
    ddf_dxdy_1 = subs(ddf_dxdy, [X,Y], [x(1),y(1)]);
    H = [ddf_ddx_1, ddf_dxdy_1; ddf_dxdy_1, ddf_ddy_1]; 
    S = inv(H); 
    while norm(J) > e && i<100000
        I = [x(i),y(i)]';
        x(i+1) = I(1)-r*S(1,:)*J';
        y(i+1) = I(2)-r*S(2,:)*J';
        i = i+1;
        J = [subs(df_dx,[X,Y], [x(i),y(i)]) subs(df_dy, [X,Y], [x(i),y(i)])]; 
        ddf_ddx_1 = subs(ddf_ddx, [X,Y], [x(i),y(i)]);
        ddf_ddy_1 = subs(ddf_ddy, [X,Y], [x(i),y(i)]);
        ddf_dxdy_1 = subs(ddf_dxdy, [X,Y], [x(i),y(i)]);
        H = [ddf_ddx_1, ddf_dxdy_1; ddf_dxdy_1, ddf_ddy_1];
        S = inv(H); 
    end
    h(i) = plot(x,y,'.c');
    set(h(i),'handlevisibility','off');
    disp(k)
end
hold off;