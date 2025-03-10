function x = update_sigma(y, beta)

Delta=(1-y)^2-4*(beta-y);
if Delta>0
    g0=1/2*(y^2);
    x1=1/2*(y-1+sqrt(Delta));
    g1=1/2*(x1-y)^2+beta*log(1+x1);
    if g1>g0
       x=0;
    else
       x=x1;
    end
else
     x=0;
end
end