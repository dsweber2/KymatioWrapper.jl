# KymatioWrapper
A simple wrapper to be able to call Kymatio commands via julia. Before
installing, you may want to set the environmental variable `PATH_TO_KYMATIO` to
a location other than your home directory (the default install). From within julia, do this via `ENV["PATH_TO_KYMATIO"] = /your/pathname/kymatio`


1. install pytorch via conda
2. download kymatio via `git clone https://github.com/kymatio/kymatio`
3. 

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
