library(iprior)
library(tidyverse)
theme_set(theme_bw())

func <- function(x) {
  1.0 * sin(x * 3 * pi) +
    0.3 * cos(x * 9 * pi) +
    0.5 * sin(x * 7 * pi)
}

n <- 1000  # no. of training points
m <- 30  # no. of inducing points
sigma_y <- 0.2  # noise

# Training data
X <- seq(-1, 1, length = n)
y <- func(X) + rnorm(n, sd = sigma_y)

my_kern <- function(x, y, lambda, sigma) {
  n <- length(x)
  m <- length(y)
  
  sqdist <- matrix(x ^ 2, nrow = n, ncol = m) +
    matrix(y ^ 2, nrow = n, ncol = m, byrow = TRUE) -
    2 * tcrossprod(x, y)
  
  return(
    lambda ^ 2 * exp(-0.5 * sqdist / sigma ^ 2)
  )
}

# Test data
X_test <- seq(-1.5, 1.5, length = 1000)
f_true <- func(X_test)

# ggplot(tibble(x = X, y = y), aes(x, y)) +
#   geom_line(data = tibble(x = X_test, y = f_true), 
#             col = "red3", size = 1) +
#   geom_point(size = 0.5) 

elbo_fn <- function(theta) {
  lambda <- exp(theta[1])
  ls <- exp(theta[2])
  X_m <- theta[-c(1, 2)]
  
  K_mm <- my_kern(X_m, X_m, lambda, ls)  # m x m matrix
  K_mm <- (K_mm + t(K_mm)) / 2 + diag(1e-3, m)
  # tmp <- eigen(K_mm)
  # U <- tmp$values
  # delta <- 1e-6 - min(U)
  # if (delta > 0) U <- U + delta
  # else U <- U + 1e-6
  # V <- tmp$vectors
  # print(U)
  # K_mm <- V %*% diag(U) %*% t(V)
  K_mn <- my_kern(X_m, X, lambda, ls)  # m x n matrix
  
  L <- t(chol(K_mm))
  A <- solve(L, K_mn) / sigma_y  # m x n matrix
  AAT <- tcrossprod(A)
  B <- diag(1, m) + AAT   # m x m
  LB <- t(chol(B))
  c <- as.numeric(solve(LB, A) %*% y / sigma_y)
  
  elbo <- 
    - n / 2 * log(2 * pi) - 
    sum(log(diag(LB))) -
    n / 2 * log(sigma_y ^ 2) -
    sum(y ^ 2) / (2 * sigma_y ^ 2) +
    sum(c ^ 2) / 2 -
    lambda ^ 2 * n / (2 * sigma_y ^ 2) +
    0.5 * sum(diag(AAT))
  
  return(- 2 * elbo)
}

init <- c(1, 1, seq(-0.4, 0.4, length = m))
elbo_fn(init)

res <- optim(init, elbo_fn, method = "L-BFGS-B", 
             lower = c(-Inf, -Inf, rep(-1, m)),
             upper = c(Inf, Inf, rep(1, m)))
# res <- nlm(elbo_fn, init)

lambda <- exp(res$par[1])
ls <- exp(res$par[2])
X_m <- res$par[-c(1, 2)]
K_mm <- my_kern(X_m, X_m, lambda, ls) 
K_mm <- (K_mm + t(K_mm)) / 2 + diag(1e-5, m)
K_mm_inv <- solve(K_mm)
K_nm <- my_kern(X, X_m, lambda, ls)
K_mn <- t(K_nm)

tmp <- K_mm + K_mn %*% K_nm / sigma_y ^ 2
tmp <- (tmp + t(tmp)) / 2 + diag(1e-5, m)
Sigma <- solve(tmp)

mu_m <- K_mm %*% Sigma %*% K_mn %*% y / sigma_y ^ 2
A_m <- K_mm %*% Sigma %*% K_mm
 
# ggplot(tibble(x = X, y = y), aes(x, y)) +
#   geom_line(data = tibble(x = X_test, y = f_true), 
#             col = "red3", size = 1) +
#   geom_point(size = 0.5) +
#   geom_vline(data = tibble(xm = X_m), aes(xintercept = xm), inherit.aes = FALSE)

ggplot(tibble(x = X, y = y), aes(x, y)) +
  geom_line(data = tibble(x = X_test, y = f_true), 
            col = "red3", size = 1) +
  # geom_point(size = 0.5) +
  geom_point(data = tibble(x = X_m, y = mu_m)) +
  coord_cartesian(ylim = c(-2, 2))

K_ss <- my_kern(X_test, X_test, lambda, ls)
K_sm <- my_kern(X_test, X_m, lambda, ls)
K_ms <- t(K_sm)

f_q <- K_sm %*% K_mm_inv %*% mu_m

ggplot(tibble(x = X, y = y), aes(x, y)) +
  geom_line(data = tibble(x = X_test, y = f_q), 
            col = "red3", size = 1) +
  # geom_point(size = 0.5) +
  geom_point(data = tibble(x = X_m, y = mu_m)) +
  coord_cartesian(ylim = c(-2, 2))












