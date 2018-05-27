function [ydata,cost] = alphaSSNE(X, labels, alpha, weight, no_dims, decay, seeder, initial_dims, perplexity)
%ALPHASSNE Performs alpha-SSNE on dataset X
%
%   mappedX = alphaSSNE(X, labels, alpha, weight, no_dims, decay, seeder, initial_dims, perplexity)
%   mappedX = alphaSSNE(X, labels, alpha, weight, initial_solution, decay, seeder,  initial_dims, perplexity)
%
%   X            ---  input data
%   labels       ---  class labels
%   alpha        ---  alpha parameter (default = 1)
%   weight       ---  tail-heaviness parameter (default = 1)
%   no_dims      ---  number of output dimensions or the initial solution matrix (default = 2)
%   decay        ---  do alpha decay (default = false)
%   seeder       ---  set the random seeder
%   initial_dims --- numver of dimensions of the input
%
% (C) Ehsan Amid, 2014
% Aalto University, Finland
% Based on the code by Laurens van der Maaten

    if ~exist('alpha', 'var') || isempty(alpha)
        alpha = 1;      % do SSNE
    end
    if ~exist('weight', 'var') || isempty(weight)
        weight = 1;     % Student t-distribution
    end
    if ~exist('labels', 'var')
        labels = [];
    end
    if ~exist('no_dims', 'var') || isempty(no_dims)
        no_dims = 2;
    end
    if ~exist('decay', 'var') || isempty(decay)
        decay = false;
    end
     if ~exist('initial_dims', 'var') || isempty(initial_dims)
        initial_dims = min(50, size(X, 2));
    end
    if ~exist('perplexity', 'var') || isempty(perplexity)
        perplexity = 30;
    end
    if ~exist('seeder', 'var')
        s = RandStream('mcg16807', 'seed','shuffle');
        seeder = s.Seed;
    end
    
    % First check whether we already have an initial solution
    if numel(no_dims) > 1
        initial_solution = true;
        ydata = no_dims;
        no_dims = size(ydata, 2);
        perplexity = initial_dims;
    else
        initial_solution = false;
    end
    
    % Normalize input data
    X = X - min(X(:));
    X = X / max(X(:));
    X = bsxfun(@minus, X, mean(X, 1));
    
    % Perform preprocessing using PCA
    if ~initial_solution
        disp('Preprocessing data using PCA...');
        if size(X, 2) < size(X, 1)
            C = X' * X;
        else
            C = (1 / size(X, 1)) * (X * X');
        end
        [M, lambda] = eig(C);
        [lambda, ind] = sort(diag(lambda), 'descend');
        M = M(:,ind(1:initial_dims));
        lambda = lambda(1:initial_dims);
        if ~(size(X, 2) < size(X, 1))
            M = bsxfun(@times, X' * M, (1 ./ sqrt(size(X, 1) .* lambda))');
        end
        X = bsxfun(@minus, X, mean(X, 1)) * M;
        clear M lambda ind
    end
    
    % Compute pairwise distance matrix
    sum_X = sum(X .^ 2, 2);
    D = bsxfun(@plus, sum_X, bsxfun(@plus, sum_X', -2 * (X * X')));
    
    % Compute joint probabilities
    P = d2p(D, perplexity, 1e-5);                                           % compute affinities using fixed perplexity
    clear D
    
    % Run alpha-SSNE
    if initial_solution
        [ydata,cost] = alphaSSNE_s(P, labels, alpha, weight, ydata, decay, seeder);
    else
        [ydata,cost] = alphaSSNE_s(P, labels, alpha, weight, no_dims, decay, seeder);
    end
    
