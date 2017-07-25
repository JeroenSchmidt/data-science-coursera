clear
Z =  [1 2 3 4 5 6 7 8 10];
B = zeros(9,10);

for Count = 1:9
  B(Count,Z(Count)) = 1;
endfor