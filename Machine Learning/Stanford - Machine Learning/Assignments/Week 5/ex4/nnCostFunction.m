function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%

% forward propagate
A1 = [ones(m,1) X]; %input layer

% first hidden layer - with 25 activation units
Z2 = A1*Theta1'; 
A2 = sigmoid(Z2);

% output layer, in this example we have 10 output digits per x \in X
A2 = [ones(m,1) A2];
Z3 = A2*Theta2';
A3 = sigmoid(Z3); %our output units h=a3 

% Create Matrix for Y values converted to binary vector representation
k = size(A3,2); % number of output units
Y = zeros(m,k);
i = sub2ind(size(Y), 1:rows(Y), y');
Y(i) = 1;

% loop accross all training examples m, 
% calculated accross K nodes using vector multiplication
for i=1:m
  J = J + -Y(i,:)*log(A3(i,:)')-(1-Y(i,:))*log(1-A3(i,:)');
endfor

%the following is vectorized
%J = sum(sum(-Y.*log(a3)-(1-Y).*log(1-a3)))

% Regularization
% Remove the Bias Coloumn 
Theta1_noBias = Theta1(1:end,2:end)(:);
Theta2_noBias = Theta2(1:end,2:end)(:);

reg_cost = lambda/(2*m)*(sum(Theta1_noBias.^2) + sum(Theta2_noBias.^2));

J = 1/m*J + reg_cost;
J



% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%




%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%                Vectorised BackPropagation                                  %%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Initail delta from output layer
delta_3 = A3-Y;

%Remove first vector col that is for the bias unit
% This error delta can be though of the error from the output layer being 
% propagated by the weights theta2 back into layer 2
delta_2  = delta_3*Theta2(:,2:end).*sigmoidGradient(Z2);

% Create Delta -> error * activation units gives us the partial wrt theta_ij 
Delta_2 = 1/m*delta_3'*A2;
Delta_1 = 1/m*delta_2'*A1;

%%%%% Theta gradients 
% Regularization of gradient, adding zero vector in i=1 to account for bias vector col
Theta1_reg = (lambda/m)*Theta1;
Theta1_reg(:,1) = zeros(size(Theta1,1),1);
Theta2_reg = (lambda/m)*Theta2;
Theta2_reg(:,1) = zeros(size(Theta2,1),1);

% Combining Delta term and Regularized Theta for gradient output
Theta2_grad = Delta_2 + Theta2_reg;
Theta1_grad = Delta_1 + Theta1_reg;

grad = [Theta1_grad(:);Theta2_grad(:)];



















% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
