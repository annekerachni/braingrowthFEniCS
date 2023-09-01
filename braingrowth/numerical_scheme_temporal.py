# Time integration numerical scheme
###################################

# Update formula for acceleration
# a = 1/(2*beta)*((u - u0 - v0*dt)/(0.5*dt*dt) - (1-2*beta)*a0)
def update_acceleration(u, u_old, v_old, a_old, beta, dt, ufl=True):

    if ufl:
        dt_ = dt
        beta_ = beta

    else:
        dt_ = float(dt)
        beta_ = float(beta)

    return (u - u_old - dt_ * v_old) / beta_ / dt_**2 - ( 1 - 2*beta_ ) / 2 / beta_ * a_old

# Update formula for velocity
# v = dt * ((1-gamma)*a0 + gamma*a) + v0
def update_velocity(a_new, u_old, v_old, a_old, gamma, dt, ufl=True):

    if ufl:
        dt_ = dt
        gamma_ = gamma

    else:
        dt_ = float(dt)
        gamma_ = float(gamma)

    return v_old + dt_ * ( (1 - gamma_) * a_old + gamma_ * a_new )

def update_fields(u, u_old, v_old, a_old, beta, gamma, dt):
    """Update fields at the end of each time step."""

    # Get vectors (references)
    u_vec, u0_vec  = u.vector(), u_old.vector()
    v0_vec, a0_vec = v_old.vector(), a_old.vector()

    # use update functions using vector arguments
    a_vec = update_acceleration(u_vec, u0_vec, v0_vec, a0_vec, beta, dt, ufl=False)
    v_vec = update_velocity(a_vec, u0_vec, v0_vec, a0_vec, gamma, dt, ufl=False)

    # Update (u_old <- u)
    v_old.vector()[:], a_old.vector()[:] = v_vec, a_vec
    u_old.vector()[:] = u.vector()

    return u_old, v_old, a_old


def avg(x_old, x_new, alpha):
    return alpha*x_old + (1-alpha)*x_new