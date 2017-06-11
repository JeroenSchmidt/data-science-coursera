# Octave

[TOC]

## Basic Operations

```
% Change prompt key
PS1('>> ');
```

```octave
disp(<var>)
```

```octave
% Example
a = pi;
disp(sprintf('2 decimals: %0.2f',a))
%% %0.2f specifies formatting of what a should be
```

```octave
% changing display format
format long
format short
```

```octave
% comma chaining
a = 1, b = 2, c =3
% using ; will supress output
```

### Vector & Matrices

```octave
% 2x3 matrix
A = [1 2; 3 4; 5 6]
```

```octave
% row vector - 1x3
v = [1 2 3]
```

```octave
% col vector - 3x1
v = [1;2;3]
```

```octave
% Generate row by increments of 0.1
v = 1:0.1:2
```

```octave
% create ones vector/matrix
ones(2,3)
```

```octave
% create zeroes vector/matrix - 1x3
zeros(1,3)
```

```octave
% create random vector/matrix - 3x3
rand(3,3)
```

```octave
% gausian dist random vector/matrix 
randn(3,3)
```

```octave
% identity matrix
eye(<size>)
```

```octave
% matrix dimensions
size(<vector>)
```

```octave
% return largest dim of vector
length(<vector>)
```

## Moving Data Around

```octave
% working dir
pwd()
% list dir
ls
```

```octave
% load strings
load('file name')
```

```octave
% show active variables
who

% more details
whos
```

```octave
% delete var
clear(<var name>)

 % delete all
 clear
```

```octave
% saving files
save('<file name>',<var>)

% compressed format
save('file_name.mat',<var>)

save file_name.txt <var> -ascii
```

### Indexing

```octave
% row 3, col 2
A(3,2)
```

```octave
% : means every element along that row/col
% return row 2
A(:,2)
```

```octave
% get all elements in rows 1 and 3
A([1 3],:)
```

```octave
% Append to the right
A = [A ; [100;101;102]]
```

```octave
% put all elemnts into a single vector
A(:)
```

```octave
% concatinating side by side
A = [1 2 ; 3 4 ; 5 6]
B = [11 12 ; 13 14 ; 15 16]
C = [A B]
% equivilant to 
C = [A,B]
>> C =
    1    2   11   12
    3    4   13   14
    5    6   15   16
    
% concatinating ontop of each other
C = [A ; B]
```

## Computing on Data

```octave
% Element wise operations -> use a .

% element wise multiplication
A .* B

% element wise division
A ./ B

% element wise squaring
A .^ 2
```

```octave
% exponent e operation
exp(<vector>)
```

```octave
% log operation
log(<vector>)
```

```octave
% Inc each element by 1
v + ones(length(v),1 )
% or 
v + 1
```

```octave
% transpose
A'
```

```octave
% get max value and index
[val, ind] = max(A)
% note, if a matrix, itll return the coloumn wise index, ie the last element is a max of 3x2 matrix -> ind=6
```

```octave
% element wise comparison
a < 3
```

```octave
% return element wise comparison positions
[row,col] = find(a < 4)
```

```octave
% create magic matrix -> each row, col and diag sum up to equal values
A = magic(3)
```

```octave
% product of all elements
prod(A)
```

```octave
% sum pf all elements
sum(A)
```

```octave
floor
ceil
```

```octave
% interesting vectorisation 
max(rand(3),rand(3))
% returns a 3x3 matrix, where each element is picked out of the 2 rand 3x3 matricies given to it
```

```octave
% coloumn wise max
% return max for each coloumn
max(A,[],1)
% the 1 indicates that it must look at the 1st dimension of max

% per row maximum
max(A,[],2)
```

```octave
% sum diag - top left to bot right
A = magic(4)
I = eye(4)
x = A .*I
ans =
   16    0    0    0
    0   11    0    0
    0    0    6    0
    0    0    0    1
sum(sum(x,))
>> 136

% sum diag - top right to bot left
sum(sum(A.*flipud(eye(9))))
```

```octave
% sudo inverse
pinv(A)
```

## Plotting Data

 

```octave
% multiple plots on one graph
plot(t,y1)
hold on;
plot(t,y2,'r') % in red
xlabel('x-label')
ylabel('y-label')
legend('plot1','plot2')
title('my plot')
% save plot
print -dpng 'plot_name.png' % can save in other formats as well
```

```octave
% seperate plots
figure(1); plot(t,y1);
figure(2); plot(t,y2);
```

```octave
%subplots
subplot(1,2,1); %divides plot a 1x2 grid, access first element
plot(t,y1);
subplot(1,2,2);
plot(t,y2);
%change axis scale, of second plot
axis([0.5 1 -1 1]) %horizontal, verctical
```

```octave
% visualise a matrix		
imagesc(A)
% or 
imagesc(A), colorbar, colormap gray;
```



## Control Statements: for, while, if 

```octave
% for loop
for i=1:10,
	%stuff
end;
```

```octave
% while, if and break
i = 1;
while i <= 5,
	%stuff
	if i == 6,
		break;
	end;
end;
```

```octave
% if 
 if <logic>,
 	%stuff
 elseif <logic>,
 	%stuff
 else
 	%stuff
 end;
```

### Functions

```octave
% load functions
% no need to, just make sure your working dir has the file
% you can also add the path
addpath('<directory to function>')
```

```octave
% functions
function <return_variables> = function_name(pass_var) 
%stuff
% returns last operation
```

==NOTE== You can return multiple values with a function

```
function [a,b] = function_name(pass_var) 
```



