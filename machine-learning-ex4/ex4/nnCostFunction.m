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
a1 = [ones(size(X, 1), 1), X];
z2 = a1 * (Theta1');
a2 = sigmoid(z2);
temp = ones(size(a2, 1), 1);
a2 = [temp, a2];
a3 = sigmoid(a2 * Theta2')
for i = 1:size(X, 1)
    for  k = 1 : num_labels
        if y(i) == k
            J = J + (-log(a3(i, k)));
        else
            J = J + (-log(1-a3(i,k)));
        end
    end
end

J = J/size(X, 1)
temp1 = Theta1(:, 2:size(Theta1, 2));
temp2 = Theta2(:, 2:size(Theta2, 2));
temp1 = temp1 .^2;
temp2 = temp2 .^2;
res = sum(temp1(:));
res = res + sum(temp2(:));
res = lambda * res /(2 * size(X, 1));
J = J + res;

delta_3 = zeros(1, size(num_labels, 1));%1x10
Delta_2 = 0;
Delta_1 = 0;
Theta1NoBias = Theta1(:, 2:size(Theta1, 2));
Theta2NoBias = Theta2(:, 2:size(Theta2, 2));
for t = 1 : m
    z2 = a1(t, :) * (Theta1'); %1x401 * 401*25 = 1x25
    a2 = sigmoid(z2);%1x25
    temp = ones(size(a2, 1), 1);%1x1
    a2 = [temp, a2];%1x26
    a3 = sigmoid(a2 * Theta2');%1x10
    for k = 1 : num_labels
        if k == y(t)
            delta_3(k) = a3(k) - 1;
        else
            delta_3(k) = a3(k);
        end
    end

    delta_2 = (delta_3 * Theta2NoBias) .* sigmoidGradient(z2); %1x25
    Delta_2 = Delta_2 + delta_3' * a2;%10*25
    Delta_1 = Delta_1 + delta_2' * a1(t, :);%25*400
end

Theta1_grad = Delta_1/m;
Theta2_grad = Delta_2/m;

Theta1_grad = Theta1_grad + (lambda/m)*[zeros(size(Theta1, 1), 1) Theta1NoBias];
Theta2_grad = Theta2_grad + (lambda/m)*[zeros(size(Theta2, 1), 1) Theta2NoBias];

% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];

end