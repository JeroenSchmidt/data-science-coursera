function idx = findClosestCentroids(X, centroids)
%FINDCLOSESTCENTROIDS computes the centroid memberships for every example
%   idx = FINDCLOSESTCENTROIDS (X, centroids) returns the closest centroids
%   in idx for a dataset X where each row is a single example. idx = m x 1 
%   vector of centroid assignments (i.e. each entry in range [1..K])
%

% Set K
K = size(centroids, 1);

% You need to return the following variables correctly.
idx = zeros(size(X,1), 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Go over every example, find its closest centroid, and store
%               the index inside idx at the appropriate location.
%               Concretely, idx(i) should contain the index of the centroid
%               closest to example i. Hence, it should be a value in the 
%               range 1..K
%
% Note: You can use a for-loop over the examples to compute this.
%


% looping through the centroid locations and using vectorisation to avoid 
% further nested loops
m = length(X);

k = 1; 
idx = ones(m,1)*k;

% Define a very large number 
shortest_distance = ones(m,1)*100000000;

for centroid = centroids'
  centroid_vec = repmat(centroid',m,1);
  distance_new = norm( X - centroid_vec , OPT="rows");
  
  closer_positions = distance_new < shortest_distance;
  
  idx(closer_positions) = k;   
  shortest_distance(closer_positions) = distance_new(closer_positions);
  
  distance_old = distance_new;
  k += 1;
endfor

idx;


% =============================================================

end

