function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta

h = X*theta;

sigmoidh = sigmoid(h);
logh = log(sigmoidh);
log_h = log(1-sigmoidh);
y_1 = 1-y;
theta_norm = theta'*theta - theta(1)*theta(1);
theta_0 = theta;
theta_0(1) = 0;

J = -(y'*logh + y_1'*log_h - 0.5*lambda*theta_norm)/m;

grad = (X'*(sigmoidh-y) + lambda*theta_0)/m;




% =============================================================

end
