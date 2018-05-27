function [ydata, cost] = alphaSSNE_s(P, labels, alpha, weight, no_dims, decay, seeder)
%ALPHASSNE_S Performs symmetric alpha-SSNE on affinity matrix P
%
%   mappedX = alphaSSNE_s(P, labels, alpha, weight, no_dims, decay, seeder)
%   
%   P            ---  adjacency matrix
%   labels       ---  class labels
%   alpha        ---  alpha parameter (default = 1)
%   weight       ---  tail-heaviness parameter (default = 1)
%   no_dims      ---  number of output dimensions or the initial solution matrix (default = 2)
%   decay        ---  do alpha decay (default = false)
%   seeder       ---  set the random seeder
%
% (C) Ehsan Amid, 2014
% Aalto University, Finland
% Based on the code by Laurens van der Maaten

    
    if ~exist('labels', 'var')
        labels = [];
    end
    if ~exist('no_dims', 'var') || isempty(no_dims)
        no_dims = 2;
    end
    if ~exist('alpha', 'var') || isempty(alpha)
        alpha = 1;      % do SSNE
    end
    if ~exist('weight', 'var') || isempty(weight)
        weight = 1;     % Student t-distribution
    end
    
    if ~exist('decay', 'var') || isempty(decay)
        decay = false;
    end
    
    if exist('seeder', 'var')
        rng(seeder);
    end
    
    % First check whether we already have an initial solution
    if numel(no_dims) > 1
        initial_solution = true;
        ydata = no_dims;
        no_dims = size(ydata, 2);
    else
        initial_solution = false;
    end
    
    % Initialize some variables
    n = size(P, 1);                                     % number of instances
    momentum = 0.5;                                     % initial momentum
    final_momentum = 0.8;                               % value to which momentum is changed
    mom_switch_iter = 1500;                             % iteration at which momentum is changed
    stop_lying_iter = 100;                              % iteration at which lying about P-values is stopped
    max_iter = 2000;                                    % maximum number of iterations
    if alpha <=1
        epsilon = 100;                                  % initial learning rate
    else
        epsilon = 2 * max(0.1,weight);
    end
    min_gain = .01;                                     % minimum gain for delta-bar-delta
    
    if decay
        a_init = 1;
        a_final = alpha;
        if alpha < 1
        delta_a = (a_init - a_final)/10;
        alpha = a_init;
        else
            alpha = a_final;
            decay = false;
        end
        a_switch_iter = 1000;
    else
        a_final = alpha;
        a_switch_iter = 0;
        momentum = 0.5;
        final_momentum = 0.8;
    end
    
    % Make sure P-vals are set properly
    P(1:n + 1:end) = 0;  
    P = 0.5 * (P + P');                                 % symmetrize P-values
    P = max(P ./ sum(P(:)), realmin);                   % make sure P-values sum to one
  
    % Initialize the solution
    if ~initial_solution
        ydata = .0001 * randn(n, no_dims);
    end
    y_incs  = zeros(size(ydata));
    gains = ones(size(ydata));
    
    % Run the iterations
    for iter=1:max_iter
        
        % Compute joint probability that point i and j are neighbors
        sum_ydata = sum(ydata .^ 2, 2);
        num = bsxfun(@plus, sum_ydata, bsxfun(@plus, sum_ydata', -2 * (ydata * ydata')));
        if weight==0
            num = exp(-num);
        else
            num = 1 ./ (1 + weight * num).^(1/weight); % Student-t distribution
        end
        
        num(1:n+1:end) = eps;  % set diagonal to zero

        Q = max(num ./ sum(num(:)), realmin);                               % normalize to get probabilities
        
        if iter < stop_lying_iter
            if alpha == 0
                L = 0.005/4*(sum(Q(:).*log(Q(:)./P(:)))*Q - Q.*log(Q./P)).*num.^weight;
            else
                L = (4*P.^alpha.*Q.^(1-alpha)- sum(P(:).^alpha.*Q(:).^(1-alpha))*Q).*num.^weight;
            end
        else
            if alpha == 0
                L = 0.005/4*(sum(Q(:).*log(Q(:)./P(:)))*Q - Q.*log(Q./P)).*num.^weight;
            else
                L = (P.^alpha.*Q.^(1-alpha)- sum(P(:).^alpha.*Q(:).^(1-alpha))*Q).*num.^weight;
            end
        end
          y_grads = -4 * (diag(sum(L, 2)) - L) * ydata;
            
        % Update the solution
        if momentum > 0 
            gains = (gains + .2) .* (sign(y_grads) ~= sign(y_incs)) ...         % note that the y_grads are actually -y_grads
                  + (gains * .8) .* (sign(y_grads) == sign(y_incs));
            gains(gains < min_gain) = min_gain;
        end
        y_incs = momentum * y_incs - epsilon * (gains .* y_grads);
        ydata = ydata - y_incs;

        ydata = bsxfun(@minus, ydata, mean(ydata, 1));
        
        % Update the momentum if necessary
        if iter == mom_switch_iter
            momentum = final_momentum;
        end
        
        if decay 
            if iter < a_switch_iter && ~rem(iter,100)
                alpha = alpha - delta_a;                    % decrease alpha
            elseif iter == a_switch_iter
                alpha = a_final;                            % set alpha to the final value
                momentum = 0.5;
                final_momentum = 0.8;
            end
        end
 
        % Print out progress
        if ~rem(iter, 10)
            if alpha ==0
                cost = (Q(:).*log(Q(:)./P(:)));
                cost(1:n+1:end) = 0;
            elseif alpha ==1
                cost = (P(:).*log(P(:)./Q(:)));
                cost(1:n+1:end) = 0;
            else
                cost = 1/alpha/(alpha-1)*(P(:).^alpha.*Q(:).^(1-alpha)-1/n^2); % a*P(:)+(a-1)*Q(:));
            end
            cost = sum(cost);
            disp(['Iteration ' num2str(iter) ': error is ' num2str(cost)]);
        end
        
        % Display scatter plot (maximally first three dimensions)
        if ~rem(iter, 10) && ~isempty(labels)
            if no_dims == 1
                scatter(ydata, ydata, 9, labels, 'filled');
            elseif no_dims == 2
                scatter(ydata(:,1), ydata(:,2), 9, labels, 'filled');
            else
                scatter3(ydata(:,1), ydata(:,2), ydata(:,3), 40, labels, 'filled');
            end
            axis tight
            axis off
            drawnow
        end
    end
    