clear;clc;close all;
syms x1 x2 x3 alpha; % symbolic
f = 2*x1^2 + x2^2 + 4*x3^2 - 5*x1 + 9*x2;
k = 1; % iteration number
X(k,:) = [9 4 2]; % initial guess
error = 100;
delta = 0.00001; % for finite difference (approximation of derivative)

disp(['Initial Guess = [' num2str(X(k,:)) ']']);
function_call = 0;

while error > 10^-4 % iteration stops when error <= 10^-3
    f_current = subs(f,{'x1' 'x2' 'x3'},X(k,:)); function_call = function_call +1; % evaluating at current point
    f_dx = subs(f,{'x1' 'x2' 'x3'},X(k,:)+[delta 0 0]); function_call = function_call +1; % evaluate at current + delta_x
    f_dy = subs(f,{'x1' 'x2' 'x3'},X(k,:)+[0 delta 0]); function_call = function_call +1; % evaluate at current + delta_y
    f_dz = subs(f,{'x1' 'x2' 'x3'},X(k,:)+[0 0 delta]); function_call = function_call +1; % evaluate at current + delta_z
    grad = [(f_dx - f_current)/delta, ...
            (f_dy - f_current)/delta, ...
            (f_dz - f_current)/delta]; % estimating gradiant using forward scheme of finite difference

    if k == 1 % newly added for conjugate gradient
        dir = -grad;
    else
        dir = -grad + (grad*grad') / (previous_grad*previous_grad') * previous_dir;
    end

    previous_grad = grad; % memorizing grad and dir
    previous_dir = dir;
    X_alpha = X(k,:) + alpha * dir; % recursive formula for steepest descent method but step size still unknown
    
    % initiate sub-problem
    f_alpha = subs(f,{'x1' 'x2' 'x3'},X_alpha); % substituting back to f to get f(alpha)
    m = 1; % iteration number of sub-problem
    A(m) = 0; % initial guess of alpha
    error_alpha = 100;
    while error_alpha > 10^-4
        if m ==1 % do not re-evaluate function at the same position (can only do this if initial guess of alpha = 0)
            f_alpha_current = f_current;
        else
            f_alpha_current = subs(f_alpha,'alpha',A(m)); function_call = function_call +1; % evaluate f(alpha) at current alpha
        end
        f_alpha_plus = subs(f_alpha,'alpha',A(m)+delta); function_call = function_call +1; % f(alpha + delta)
        f_alpha_minus = subs(f_alpha,'alpha',A(m)-delta); function_call = function_call +1; % f(alpha - delta)
        f_alpha_first = (f_alpha_plus - f_alpha_current)/delta; % estimating f'(alpha)
        f_alpha_second = (f_alpha_plus - 2*f_alpha_current + f_alpha_minus)/(delta)^2; % f''(alpha)
        A(m+1) = A(m) - f_alpha_first/f_alpha_second; % 1D Newton to find next alpha
        error_alpha = abs(A(m+1)-A(m));
        m=m+1;
        disp(A)
    end
    

    X(k+1,:) = subs(X_alpha,'alpha',A(end)); % updating optimal alpha back to the recursive formula
    error = norm(X(k+1,:)-X(k,:));
    disp(['Iter ' num2str(k) ', X = [' num2str(X(k+1,:)) '], error = ' num2str(error) ,', FC = ', num2str(function_call)]);
    
    k=k+1;
end