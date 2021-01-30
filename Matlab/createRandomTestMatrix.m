filename = "";      %change filename

n = 5;
k = 20; %percentage

out = zeros(n);
out(randperm(n^2, ceil(n^2*k/100))) = 1;

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

shape = [n, n];
vector = horzcat(shape, flatten);

dlmwrite(filename, vector,  'delimiter', '\n');
