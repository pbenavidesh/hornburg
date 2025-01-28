# SVRL1_MAPE derivation

plane: $y - f(x) = 0$


Distance from the plane: $\frac{f(x) - y}{||\omega^*||}$


**Primal Problem**

Given

$$
\begin{align}
% min D = 1/2 w^2 + C sum(E + E*)
\min_{\omega, b, \xi} \;\; & \frac{1}{2} ||\omega||^2 + C\sum_{k = 1}^N \left( \xi + \xi^* \right) \\
% 100(y - f(x) <= e + E)/y
\text{S. t. } \;\; & 100\frac{y_k - \langle \omega, \phi(x_k) \rangle - b}{y_k} \leq \epsilon + \xi_k & (1)\\
% 100(y - f(x) <= e + E*)/y
              & 100\frac{\langle \omega, \phi(x_k) \rangle + b - y_k}{y_k} \leq \epsilon + \xi^*_k\\
% E, E* >= 0
              & \xi_k, \xi^*_k \geq 0 \\
\end{align}
$$

<font size="1"> For $y_k \neq 0$ </font>


**Lagrangian**

$$
\begin{align}
% L
\mathcal{L}(\omega, b, \xi, \alpha, \alpha^*, \eta, \eta^*) = &
% 1/2 w^2
\frac{1}{2} ||\omega||^2 + C\sum_{k = 1}^N \left( \xi + \xi^* \right) -
% a((e + E)y/100 -y + f(x))
\sum_{k=1}^N \alpha_k \left[y_k\frac{\epsilon + \xi_k}{100} - y_k + \langle \omega, \phi(x_k) \rangle + b \right] \\
% a((e + E)y/100  - f(x) + y)
& - \sum_{k=1}^N \alpha^*_k \left[ y_k\frac{\epsilon + \xi_k^*}{100} + y_k - \langle \omega, \phi(x_k) \rangle - b  \right] -
% n E + n* E*
\sum_{k=1}^N \left( \eta_k \xi_k + \eta^*_k \xi^*_k \right) & (2)
\end{align}
$$

**Primal Feasibility**

$$
\begin{equation}
    \epsilon + \xi_k \geq 100\frac{y_k - \langle \omega, \phi(x_k) \rangle - b}{y_k}\\
    \epsilon + \xi_k \geq 100\frac{\langle \omega, \phi(x_k) \rangle + b - y_k}{y_k}\\
    \xi_k, \xi^*_k \geq 0
\end{equation}
$$

**Dual feasibility**

$$
\begin{align}
    \alpha_k, 
    \alpha^*_k,
    \eta_k, 
    \eta^*_k, \geq 0
\end{align}
$$

**Stationarity**

<!--  omega gradient -->
- $\nabla_{\omega} \mathcal{L}$

$$
\begin{align}
    \nabla_{\omega} \mathcal{L} = \;\; & \omega - 
    \sum^N_{k = 1} \alpha_k \phi(x_k) + \sum^N_{k = 1} \alpha^*_k \phi(x_k) =
    \omega - \sum_{k=1}^N \left( \alpha_k \phi(x_k) - \alpha_k^* \phi(x_k) \right) = 
    \omega - \sum_{k=1}^N \left( \alpha_k - \alpha_k^* \right)\phi(x_k) = 0 & (3a)\\
    & \boxed{\omega = \sum_{k=1}^N \left( \alpha_k - \alpha_k^* \right)\phi(x_k)}
\end{align}
$$

<!-- derivative over b -->
- $\frac{\partial \mathcal{L}}{\partial b}$

$$
\begin{align}
    \frac{\partial \mathcal{L}}{\partial b} = \;\; & - \sum_{k=1}^N \alpha_k + \sum_{k=1}^N \alpha_k^* = 
    \boxed{\sum_{k=1}^N \left( \alpha_k^* - \alpha_k \right) = 0} & (3b)
\end{align}
$$

<!-- derivative over xi -->
$\frac{\partial \mathcal{L}}{\partial \xi_k}$

$$
\begin{align}
    & \frac{\partial \mathcal{L}}{\partial \xi_k} = C - \frac{\alpha_k}{100}y_k - \eta_k = 0 & (3c)
\end{align}
$$

<!-- derivative over xi* -->
$\frac{\partial \mathcal{L}}{\partial \xi^*_k}$

$$
\begin{align}
   & \frac{\partial \mathcal{L}}{\partial \xi^*_k} = C - \frac{\alpha_k^*}{100}y_k - \eta_k^* = 0 & (3d)
\end{align}
$$

**Dual derivation**

Substituting on (2)

$$
\begin{align}
% dual function
\mathcal{D}(\alpha, \eta, \eta^*) = &
% 1\2 omega^2
\underbrace{\frac{1}{2}\sum_{k, l = 1}^N \left( \alpha_k - \alpha_k^* \right)\phi(x_k) \left( \alpha_l - \alpha_l^* \right)\phi(x_l)}_{a}
% C (xi + xi*)
+ \underbrace{C\sum_{k = 1}^N \left( \xi + \xi^* \right)}_{b}
% e(a + a*)
- \underbrace{\sum_{k=1}^N \frac{\epsilon}{100} (\alpha_k + \alpha_k^*)y_k}_{c}
% a xi + a* xi*
- \underbrace{\frac{1}{100} \sum_{k=1}^N ( \alpha_k \xi_k + \alpha_k^* \xi_k^* )y_k}_{d}
% sum_{k, l} a_k (a - a*) phi phi_k
- \underbrace{\sum_{k, l=1}^N \alpha_k \left( \alpha_l - \alpha_l^* \right)\phi(x_l)\phi(x_k)}_{e}
% sum_{k, l} a_k* (a - a*) phi phi_k
+ \underbrace{\sum_{k, l=1}^N \alpha_k^* \left( \alpha_l - \alpha_l^* \right)\phi(x_l)\phi(x_k)}_{f}
% -b(a - a*)
-\underbrace{b\sum_{k=1}^N \left( \alpha_k - \alpha_k^* \right)}_{g}
% y_k(a - a*)
+ \underbrace{\sum_{k=1}^N y_k \left( \alpha_k - \alpha_k^* \right)}_{h}
- \underbrace{\sum_{k=1}^N \left( \eta_k \xi_k + \eta^*_k \xi^*_k \right)}_{i} & (4)\\
\end{align} 
$$

Rearranging (4) and substituting (3b) for each $k$ on (4g)

$$
\begin{align}
%     dual function
    \mathcal{D}(\alpha, \eta, \eta^*) = &
%     1\2 omega^2
    \underbrace{\frac{1}{2}\sum_{k, l = 1}^N \left( \alpha_k - \alpha_k^* \right) \left( \alpha_l - \alpha_l^* \right)\phi(x_l) \phi(x_k)}_{a'} 
%     omega^2
    - \underbrace{\sum_{k, l = 1}^N \left( \alpha_k - \alpha_k^* \right) \left( \alpha_l - \alpha_l^* \right)\phi(x_l) \phi(x_k)}_{e + f}
%     C - ay/100 - n
   + \underbrace{ \sum_{k=1}^N \left( C - \frac{\alpha_k}{100}y_k - \eta_k \right)\xi}_{b1 + d1 + i1}
%    C - a*y/100 - n*
    + \underbrace{ \sum_{k=1}^N \left( C - \frac{\alpha_k^*}{100}y_k - \eta_k^* \right)\xi}_{b2 + d2 + i2}
%     e(a + a*)y/100
- \underbrace{\frac{\epsilon}{100} \sum_{k=1}^N (\alpha_k + \alpha_k^*)y_k}_{c}
%     y_k(a - a*)
+ \underbrace{\sum_{k=1}^N y_k \left( \alpha_k - \alpha_k^* \right)}_{h} & (5)\\
\end{align}
$$

From (3c) and (3d), we can observe that each term on (b1 + d1 + i1) and (b2 + d2 + i2) is equal to 0, then both terms vanish.

$$
\begin{align}
%     dual function
    \mathcal{D}(\alpha,\alpha^* \eta, \eta^*) = &
%     -1\2 omega^2
    \underbrace{-\frac{1}{2}\sum_{k, l = 1}^N \left( \alpha_k - \alpha_k^* \right) \left( \alpha_l - \alpha_l^* \right)\phi(x_l) \phi(x_k)}_{a'-(e + f)}     
%     e(a + a*)y/100
- \underbrace{ \frac{\epsilon}{100} \sum_{k=1}^N y_k(\alpha_k + \alpha_k^*)}_{c}
%     y_k(a - a*)
+ \underbrace{\sum_{k=1}^N y_k \left( \alpha_k - \alpha_k^* \right)}_{h} & (6)
\end{align}
$$

**Dual Problem**

From the complementary Slackness conditions, (3b) and (10)

$$
\begin{align}
%     dual function
    \max_{\alpha, \alpha^*, \eta, \eta^*} \mathcal{D}(\alpha, \alpha^*, \eta, \eta^*) = &
%     -1\2 omega^2
    -\frac{1}{2}\sum_{k, l = 1}^N \left( \alpha_k - \alpha_k^* \right) \left( \alpha_l - \alpha_l^* \right)\phi(x_l) \phi(x_k)    
%     e(a + a*)
- \frac{\epsilon}{100} \sum_{k=1}^N (\alpha_k + \alpha_k^*)y_k
%     y_k(a - a*)
+ \sum_{k=1}^N y_k \left( \alpha_k - \alpha_k^* \right)\\
\text{S. t. } & \sum_{k=1}^N \left( \alpha_k^* - \alpha_k \right) = 0 & (7)\\
& 0 \leq \alpha_k \leq \frac{100C}{y_k}; \;\; 0 \leq \alpha_k^* \leq \frac{100C}{y_k}
\end{align}
$$

Substituting $\phi(x_l)\phi(x_k)$ with a kernel and minimizing the negative

$$
\begin{align}
%     dual function
    \min_{\alpha, \alpha^*, \eta, \eta^*} \mathcal{D}(\alpha, \alpha^*, \eta, \eta^*) = &
%     -1\2 omega^2
    \frac{1}{2}\sum_{k, l = 1}^N \left( \alpha_k - \alpha_k^* \right) \left( \alpha_l - \alpha_l^* \right)K(x_l, x_k)    
%     e(a + a*)
+ \frac{\epsilon}{100} \sum_{k=1}^N (\alpha_k + \alpha_k^*)y_k
%     y_k(a - a*)
- \sum_{k=1}^N y_k \left( \alpha_k - \alpha_k^* \right)\\
\text{S. t. } & \sum_{k=1}^N \left( \alpha_k^* - \alpha_k \right) = 0 & (8)\\
& 0 \leq \alpha_k \leq \frac{100C}{y_k}; \;\; 0 \leq \alpha_k^* \leq \frac{100C}{y_k}
\end{align}
$$

**Complementary Slackness condition**

$$
\begin{align}
% a(e + E - y + f(x))
& \alpha_k \left[ y_k\frac{\epsilon + \xi_k}{100} - y_k + \langle \omega, \phi(x_k) \rangle + b \right] = 0, & \eta_k \xi_k = 0 \\
% a(e + E - f(x) + y)
& \alpha_k^*\left[ y_k\frac{\epsilon + \xi_k^*}{100} + y_k - \langle \omega, \phi(x_k) \rangle - b \right] = 0, & \eta_k^* \xi_k^* = 0
\end{align}
$$

- $\mathbf{\alpha_k, \alpha_k^*  = 0}$

if $\alpha_k = 0 \rightarrow \;\; \eta_k = C \rightarrow \;\; \eta_k \xi_k = 0 \rightarrow \;\; \xi_k = 0$

$$
\begin{align}
    \alpha_k\left( y_k\frac{\epsilon + \xi_k}{100} - y_k + f(x_k) \right) = 0  \rightarrow \frac{\epsilon}{100}y_k - y_k + f(x_k) \geq 0\\
    y_k - f(x_k) \leq \frac{\epsilon}{100}y_k \longleftrightarrow
    100\frac{y_k - f(x_k)}{y_k} \leq \epsilon
\end{align}
$$

if $\alpha_k^* = 0 \rightarrow \;\; \eta_k^* = C \rightarrow \;\; \eta_k^* \xi_k^* = 0 \;\; \rightarrow \xi_k^* = 0$

$$
\begin{align}
    \alpha_k^*\left( y_k\frac{\epsilon + \xi_k^*}{100} + y_k - f(x_k) \right) = 0  \rightarrow \frac{\epsilon}{100}y_k + y_k - f(x_k) \geq 0\\
     y_k - f(x_k) \geq -\frac{\epsilon}{100}y_k \longleftrightarrow 
    100\frac{y_k - f(x_k)}{y_k} \geq -\epsilon
\end{align}
$$

- $\mathbf{\alpha_k, \alpha_k^* = \frac{100C}{y_k}}$


If $\xi_k \geq 0 \;\; \rightarrow \;\; \eta_k = 0 \;\; \rightarrow \;\; \alpha_k = \frac{100C}{y_k}$
$$
\begin{align}
     & - y_k + f(x_k) \geq -y_k\frac{\epsilon + \xi_k}{100}  \;\; \rightarrow\\ 
     &\;\; y_k - f(x_k) \leq y_k\frac{\epsilon + \xi_k}{100} \longleftrightarrow 100\frac{y_k - f(x_k)}{y_k} \geq \epsilon + \xi_k
\end{align}
$$

If $\xi_k^* \geq 0 \;\; \rightarrow \;\; \eta_k^* = 0 \;\; \rightarrow \;\; \alpha_k^* = \frac{100C}{y_k}$
$$
\begin{align}
     y_k - f(x_k) \geq - y_k\frac{\epsilon + \xi_k^*}{100} \longleftrightarrow 100\frac{y_k - f(x_k)}{y_k} \geq -\epsilon - \xi_k^*
\end{align}
$$

- $\mathbf{\alpha_k, \alpha_k^* = \frac{100C}{y_k}}$


If $\xi_k \geq 0 \;\; \rightarrow \;\; \eta_k = 0 \;\; \rightarrow \;\; \alpha_k = \frac{100C}{y_k}$
$$
\begin{align}
     & - y_k + f(x_k) \geq -y_k\frac{\epsilon + \xi_k}{100}  \;\; \rightarrow\\ 
     &\;\; y_k - f(x_k) \leq y_k\frac{\epsilon + \xi_k}{100} \longleftrightarrow 100\frac{y_k - f(x_k)}{y_k} \geq \epsilon + \xi_k
\end{align}
$$

If $\xi_k^* \geq 0 \;\; \rightarrow \;\; \eta_k^* = 0 \;\; \rightarrow \;\; \alpha_k^* = \frac{100C}{y_k}$
$$
\begin{align}
     y_k - f(x_k) \geq - y_k\frac{\epsilon + \xi_k^*}{100} \longleftrightarrow 100\frac{y_k - f(x_k)}{y_k} \geq -\epsilon - \xi_k^*
\end{align}
$$


From here, it can be derived that $y_k$ must be positive valued.

The greater the difference between y_k and 100C, the less contribution it has to the model.

- $\mathbf{0 < \alpha_k < \frac{100}{y_k}C, \;\;  0 < \alpha_k^* < \frac{100}{y_k}C}$


$$
\begin{align}
% y - f = e
    \text{if } \xi_k = 0 \;\; \rightarrow \;\; \eta_k \geq 0 \;\; \rightarrow \;\; 0 < \alpha_k < \frac{100C}{y_k} \;\; \rightarrow \;\; \\ y_k - f(x_k) = \frac{\epsilon}{100}y_k \longleftrightarrow 100\frac{y_k - f(x_k)}{y_k} = \epsilon\\
% y - f = - e
    \text{if } \xi_k^* = 0 \;\; \rightarrow \;\; \eta_k^* \geq 0 \;\; \rightarrow \;\; 0 < \alpha_k^* < \frac{100C}{y_k} \;\; \rightarrow \;\; \\y_k - f(x_k) = -\frac{\epsilon}{100}y_k \longleftrightarrow 100\frac{y_k - f(x_k)}{y_k} = -\epsilon
\end{align}
$$

# SVRL1_MAPE Active set

From the previous problem, it can be notice that 2N variables are necessary to solve the problem, where N is the number of examples. If N is big, the computation can become very expensive. Therefore, a new set of variables is proposed to decrease the number of variables to N.

|                    | Over the tube                | On the upper border               | Inside the tube | On the lower bound               | Above the tube                |
|:------------------:|:----------------------------:|:---------------------------------:|:---------------:|:-------------------------------:|:-----------------------------:|
| $\alpha$           | $\frac{100}{y_k}C$           | $> 0, < \frac{100}{y_k}C$         | 0               | 0                               | 0                             |
| $\xi$              | $\geq 0$                     | 0                                 | 0               | 0                               | 0                             |
| $\eta$             | 0                            | $C$                               | $C$             | $C$                             | $C$                           |
| $\alpha^*$         | 0                            | 0                                 | 0               | $> 0, < \frac{100}{y_k}C$       | $\frac{100}{y_k}C$            |
| $\xi^*$            | 0                            | 0                                 | 0               | 0                               | $\geq 0$                      |
| $\eta^*$           | $C$                          | $C$                               | $C$             | $C$                             | 0                             |

Table 1. The asterisk variables do not interfere on the opposite variable

From Table 1 we can define,

$
\begin{align} 
     & \beta_k = \alpha_k - \alpha_k^* \quad \rightarrow -\frac{100}{y_k}C \leq \beta \leq \frac{100}{y_k}C &
     \\  
     \\
     & |\beta_k| = \alpha_k +\alpha_k^* \quad \rightarrow 0 \leq |\beta| \leq \frac{100}{y_k}C &
     \\
     \\
     & \xi_k = \xi_k + \xi_k^* \quad \rightarrow \xi_k \geq 0 &
     \\  
     \\
     & \eta_k = \eta_k + \eta_k^* \quad \rightarrow  \eta_k = C \otimes \eta_k = 2C&
\end{align}
$

<font size="1"> For $y_k \geq 0$ </font>

And substituting on (8)

$$
\begin{align}
    %     dual function
        \min_{\beta, \eta} \mathcal{D}(\beta, \eta) = &
    %     -1\2 omega^2
        \frac{1}{2}\sum_{k, l = 1}^N \beta_k \beta_l K(x_l, x_k)  
    %     e(a + a*)
    + \frac{\epsilon}{100} \sum_{k=1}^N |\beta_k|y_k
    %     y_k(a - a*)
    - \sum_{k=1}^N y_k \beta_k \\
    \text{S. t. } & \sum_{k=1}^N  \beta_k = 0 & (9)\\
    & |\beta_k|  \leq \frac{100}{y_k}C \longleftrightarrow \beta_k \leq \frac{100}{y_k}C, \quad \beta_k \geq -\frac{100}{y_k}C\\
\end{align}
$$

**Slackness condition**

$$
\begin{align}
    % a(e + E - y + f(x))
    & \beta_k \left[ y_k\frac{\epsilon + \xi_k}{100} - y_k + \langle \omega, \phi(x_k) \rangle + b \right] = 0, & \beta \geq 0 
    \\
    % a(e + E - f(x) + y)
    & \beta_k\left[ y_k\frac{\epsilon + \xi_k}{100} + y_k - \langle \omega, \phi(x_k) \rangle - b \right] = 0, & \beta \leq 0
    \\
    & \eta_k \xi_k = 0
\end{align}
$$

**Prediction**

$$
b = \left\{
\begin{array}{ll}
& y_k\left[1 - \frac{\epsilon}{100}\right] - f(x) 
\\
& & 0 < |\beta| < 100\frac{C}{y_k}\\
& y_k\left[1 + \frac{\epsilon}{100}\right] - f(x)
\end{array}
\right.
$$

$$
\omega = \sum_{k=1}^N \beta_k\phi(x_k)
$$

**Vector notation**

Lets define

$$\mathbf{\beta} = [\beta_1, \beta_2,  ..., \beta_{N - 1}, \beta_N]^T$$

$$|\mathbf{\beta}| = \left[ |\beta_1|, |\beta_2|,  ..., |\beta_{N - 1}|, |\beta_N| \right]^T$$

$$
\mathcal{K} = \left[
\begin{array}{ll}
k(x_1, x_1) & k(x_1, x_2) & ... k(x_1, x_N)\\
k(x_2, x_1) & k(x_2, x_2) & ... k(x_2, x_N)\\
.\\
.\\
k(x_M, x_1) & k(x_M, x_2) & ... k(x_M, x_N)\\
\end{array}
\right]
$$

$$\mathbf{y} = [y_1, y_2,..., y_{N - 1}, y_N]^T$$

$$ \mathbf{\frac{1}{y}} = \left[\frac{1}{y_1}, \frac{1}{y_2}, ..., \frac{1}{y_{N-1}}, \frac{1}{y_{N}}\right]^T$$

(9) can be written as

$$
\begin{align}
%     dual function
    \min_{\beta, \eta} \mathcal{D}(\beta, \eta) = &
%     -1\2 omega^2
    \frac{1}{2} \beta^T \mathcal{K} \beta  
%     e(a + a*)
+ \frac{\epsilon}{100}  \mathbf{y}^T|\beta|
%     y_k(a - a*)
-  \mathbf{y}^T \beta
\\
\text{S. t. } &   \mathbf{1_{v}}^T\beta = \mathbf{0_v} & (10)\\
& \beta \preceq \mathbf{\frac{1}{y}}100C, \quad \beta \succeq - \mathbf{\frac{1}{y}}100C
\end{align}
$$