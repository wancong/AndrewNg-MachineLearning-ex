function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
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
C_vec = [0.01 0.03 0.1 0.3 1 3 10 30]';
sigma_vec = [0.01 0.03 0.1 0.3 1 3 10 30]';
mean_min = 1;
ans_C = 0;
ans_sigma = 0;
for i = 1:length(C_vec) 
	for j = 1:length(sigma_vec) 
		C = C_vec(i); 
		sigma = sigma_vec(j); 
		
		model= svmTrain(X, y, C, @(x1, x2) gaussianKernel(x1, x2, sigma)); 
		predictions = svmPredict(model, Xval); 
		mean_tmp = mean(double(predictions ~= yval));
		if mean_tmp < mean_min
			mean_min = mean_tmp;
			ans_C = C;
			ans_sigma = sigma;
		end	
	end
end 

C = ans_C;
sigma = ans_sigma;





% =========================================================================

end
