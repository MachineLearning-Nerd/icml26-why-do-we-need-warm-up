# Source audit

Section 3.2 defines the proxy as the norm of gradients evaluated on consecutive minibatches and iterates divided by the update norm. The batch changes at the same time as the weights, so its numerator contains both curvature-induced change and minibatch noise.
