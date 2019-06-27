% The 1st example of GPML toolbox

% The dataset
x = gpml_randn(0.8, 20, 1);                 % 20 training inputs
y = sin(3*x) + 0.1*gpml_randn(0.9, 20, 1);  % 20 noisy training targets
xs = linspace(-3, 3, 61)';                  % 61 test inputs

% Specify the mean, cov, likelihood
meanfunc = [];                    % empty: don't use a mean function
covfunc = @covSEiso;              % Squared Exponental covariance function
likfunc = @likGauss;              % Gaussian likelihood

% The hyperparameter struct
hyp = struct('mean', [], 'cov', [0 0], 'lik', -1);
% NOTE 1: the mean function is empty, so takes no parameters.
% NOTE 2: The covariance function is covSEiso, the squared exponential
% with isotropic distance measure, which takes two parameters.
% NOTE 3: The Gaussian likelihood function has a single parameter, 
% which is the log of the noise standard deviation.

% Optimization
hyp2 = minimize(hyp, @gp, -100, @infGaussLik, meanfunc, covfunc, likfunc, x, y);
% The minimize function minimizes the negative log marginal likelihood,
% which is returned by the gp function.
% The minimize function is allowed a computational budget of 
% 100 function evaluations.

% Result:
% length-scale is exp(-0.6352)=0.53
% signal std dev is exp(-0.1045)=0.90
% the noise std dev is exp(-2.3824)=0.092

% Finished hyperparameter optimization!

% Let us make a prediction with above parameters!
[mu s2] = gp(hyp2, @infGaussLik, meanfunc, covfunc, likfunc, x, y, xs);

% PLOT
f = [mu+2*sqrt(s2); flipdim(mu-2*sqrt(s2),1)];
fill([xs; flipdim(xs,1)], f, [7 7 7]/8)
% the 95% confidence area
hold on
plot(xs, mu,'b')  % predictive mean
plot(x, y, 'r+')  % training data