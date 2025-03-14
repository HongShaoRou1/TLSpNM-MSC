function [X] = TLSpNM_Shrink(X, lambda, mode)

sX = size(X);

if mode == 3
    Y = shiftdim(X, 1);
else
    Y = X;
end

Yhat = fft(Y,[],3);

if mode == 3
    n3 = sX(1);
    m = min(sX(2), sX(3));
else
    n3 = sX(3);
    m = min(sX(1), sX(2));
end


for i = 1:n3
    [uhat,shat,vhat] = svd(full(Yhat(:,:,i)),'econ');   
    for j = 1:m
        shat(j,j) = update_sigma(shat(j,j), lambda);
    end        
    Yhat(:,:,i) = uhat*shat*vhat';
end

Y = ifft(Yhat,[],3);
if mode == 3
    X = shiftdim(Y, 2);
else
    X = Y;
end
end




