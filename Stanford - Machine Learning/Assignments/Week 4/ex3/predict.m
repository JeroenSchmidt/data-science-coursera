function p = predict(Theta1, Theta2, X)
%PREDICT Predict the label of an input given a trained neural network
%   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
%   trained weights of a neural network (Theta1, Theta2)

% Useful values
m = size(X, 1);
num_labels = size(Theta2, 1);

% You need to return the following variables correctly 
p = zeros(size(X, 1), 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Complete the following code to make predictions using
%               your learned neural network. You should set p to a 
%               vector containing labels between 1 to num_labels.
%
% Hint: The max function might come in useful. In particular, the max
%       function can also return the index of the max element, for more
%       information see 'help max'. If your examples are in rows, then, you
%       can use max(A, [], 2) to obtain the max for each row.
%

% =================== NOTE TO SELF ============================

% X is a matrix of mutiple inputs, ie each row vector is of length 400 wihch is 
% just a 20x20 image thats been reduced into a single row 

% this implementation could use a for loop... but for understanding purposes,  
% I have left the sequencial calculations of the neural nets like this

% =============================================================

% Concat the bias input for every x \in X
a1 = [ones(m,1) X]; %input layer

% first hidden layer - with 25 activation units
z2 = a1*Theta1'; 
a2 = sigmoid(z2);

% output layer, in this example we have 10 output digits per x \in X
a2 = [ones(m,1) a2];
z3 = a2*Theta2';
a3 = sigmoid(z3);

% Find max
% For each input x \in X we produced an output hypothesis with 10 output units
% i_max is returning the index of the highest output unit per x
[p_max, i_max]=max(a3, [], 2);

p = i_max;


% =========================================================================


end
