Generic paper2code implementation of "A Fourier Perspective on Model Robustness in
Computer Vision" 

## 2D Fourier Transform (for Images)

Forward 2D Fourier Transform:

$$
F(u,v) = \sum_{x=0}^{M-1} \sum_{y=0}^{N-1} f(x,y) \, e^{-j 2 \pi \left( \frac{ux}{M} + \frac{vy}{N} \right)}
$$

- \(f(x,y)\) = image intensity at pixel \((x, y)\)  
- \(F(u,v)\) = frequency representation  
- \(M, N\) = image width and height  

Inverse 2D Fourier Transform:

$$
f(x,y) = \frac{1}{MN} \sum_{u=0}^{M-1} \sum_{v=0}^{N-1} F(u,v) \, e^{j 2 \pi \left( \frac{ux}{M} + \frac{vy}{N} \right)}
$$
