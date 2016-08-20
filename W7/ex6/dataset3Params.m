function [C, sigma] = dataset3Params(X, y, Xval, yval)
%EX6PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = EX6PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.3;

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%
err_min = inf;
c_sigma_val = [0.01 0.03 0.1 0.3 1 3 10 30];

for iC = c_sigma_val
  for iSigma = c_sigma_val
    %Train the model using svmTrain with X, y, a value for C, and the gaussian kernel using a value for sigma
    model = svmTrain(X, y, iC, @(x1, x2) gaussianKernel(x1, x2, iSigma));
    %Compure the predictions for the validation set using svmPredict() with model and Xval.
    err = mean(double(svmPredict(model, Xval) ~=yval));
    %Compute the error between your predictions and yval.
    if( err <= err_min )
      %When you find a new minimum error, save the C and sigma values that were used.
      C = iC;
      sigma = iSigma;
      err_min = err;
    end
  end
 end
 
    






% =========================================================================

end
