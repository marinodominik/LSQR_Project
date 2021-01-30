n = 9;
k = 100; %percentage

out = zeros(n);
out(randperm(n^2, ceil(n^2*k/100))) = 1;

vec = randi(100, sqrt(n));
vec = reshape(vec.', 1, [])

for a = 1:n
  for b = 1:n
    if out(a, b) == 1
       r = randi(100)
       out(a, b) = r;
    end
  endfor
endfor

flatten = reshape(out.', 1, []);

%inv_matrix = inv(out); % check if A matrix is invertable

shape_matrix = [n, n];
matrix = horzcat(shape_matrix, flatten);

shape_vector = [n, 1]
vector = horzcat(shape_vector, vec);

filename_matrix = strcat("matrix", num2str(n), "x", num2str(n), "p", num2str(k), ".txt");      %change filename
filename_vector = strcat("vector", num2str(n),"x1p", num2str(k),".txt");

dlmwrite(filename_matrix, matrix,  'delimiter', '\n');
dlmwrite(filename_vector, vector,  'delimiter', '\n');

