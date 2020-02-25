# KymatioWrapper
A simple wrapper to be able to call Kymatio commands via julia. If the build
isn't working on it's own, you may want to run the necessary commands on the
command line. In the following, substitute root for the path to your conda
environment (the value of `Conda.ROOTENV`):
```
> root/bin/conda install pytorch torchvision -c pytorch
> root/bin/pip install kymatio
```

# 1D Example
```
	J=6; Q=16; T = 2^13
	x = zeros(T, 5, 3);
	t = (range(0,2π,length=T));
	for (ii, ω)=enumerate(floor.(Int, range(1,150,length=5))), jj=1:3
		x[:,ii,jj] = sin.(ω/jj .* t) + cos.(100*ω/jj .* t.^2)
	end
	s = Scattering(T, Q=16)
	sx = s(x)
	meta = compute_meta_scattering(s)
	sx[:, meta.order .==1,1,1]
```
## 2D Example
```
	
```
