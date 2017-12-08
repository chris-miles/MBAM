import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.integrate import odeint
import numdifftools as nd

# ODEs for C(t), P(t) for Mechaelis Menten
def dydt(y, t,theta0,theta1,theta2):
    return [-(theta2+theta1)*y[0]+theta0*(0.25-y[0])*(1.0-y[0]-y[1]), theta2*y[0]];

# returns P(5),P(10),P(15) for as model "outputs" for Mechaelis Menten
def y(theta0,theta1,theta2):
    theta0 = np.exp(theta0);
    theta1 = np.exp(theta1);
    theta2 = np.exp(theta2);

    ts = np.linspace(0, 20, 201) # t values to sim over
    y0 = [0, 0] # initial values
    thetavals = (theta0,theta1,theta2)
    outvals = odeint(dydt, y0, ts, args=thetavals) # integrate ODE
    return (outvals[50,1], outvals[100,1], outvals[150,1]) # return y(5),y(10),y(15)


## sweep over parameter values to see resulting y values
paramsweep = np.linspace(-10, 10, 15) # sweep over theta_i = -10...10, for i=0,1,2
paramcombos = np.array(np.meshgrid(paramsweep, paramsweep, paramsweep)).T.reshape(-1,3) # reshape to iterate nicely

sweepsize = np.shape(paramcombos)[0]
yvalarray = np.empty([sweepsize,3]); # generate all combos of parameters
# sim over all different parameter values
for i,p in enumerate(paramcombos):
    theta0,theta1,theta2 = p;
    yvals = y(theta0,theta1,theta2);
    yvalarray[i,:] = yvals;


# plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_trisurf(yvalarray[:,0],yvalarray[:,1],yvalarray[:,2],shade=True,linewidth=0)

ax.set_xlabel(r'$y(5)$')
ax.set_ylabel(r'$y(10)$')
ax.set_zlabel(r'$y(15)$')
#plt.show()




########################
from geodesic import geodesic, InitialVelocity
# Try to do MBAM


## define slightly different version where input is a single vector of thetavals
def yy(thetavals):
    theta0 = np.exp(thetavals[0]);
    theta1 = np.exp(thetavals[1]);
    theta2 = np.exp(thetavals[2]);

    ts = np.linspace(0, 20, 101) # t values to sim over
    y0 = [0, 0] # initial values
    thetavals = (theta0,theta1,theta2)
    outvals = odeint(dydt, y0, ts, args=thetavals) # integrate ODE
    return np.array([outvals[25,1], outvals[50,1], outvals[75,1]]) # return y(5),y(10),y(15)


# jacobian of J
J=nd.Jacobian(yy)

e1 = np.array([1,0,0])
e2 = np.array([0,1,0])
e3 = np.array([0,0,1])

def finitediff(x,v):
    h = 1e-2;
    return (yy(x+h*v) - yy(x-h*v))/(2*h)

def jac(x):
    f1 = finitediff(x,e1);
    f2 = finitediff(x,e2);
    f3 = finitediff(x,e3);
    j= np.array([f1,f2,f3]).T;
    return j


# Directional second derivative
def secondderiv(x,v):
    h = 1e-3
    return (yy(x + h*v) + yy(x - h*v) - 2*yy(x))/h/h
    
    
# Choose starting parameters
thetainit = np.array([1.0, 0.5, 1.5])
vinit = InitialVelocity(thetainit, J, secondderiv)

raw_input("Press Enter to continue...")


# Callback function used to monitor the geodesic after each step
def callback(geo):
    # Integrate until the norm of the velocity has grown by a factor of 100
    # and print out some diagnotistic along the way
    print("Iteration: %i, tau: %f, |v| = %f" %(len(geo.vs), geo.ts[-1], np.linalg.norm(geo.vs[-1])))
    return np.linalg.norm(geo.vs[-1]) < 10.0

# Construct the geodesic
# It is usually not necessary to be very accurate here, so we set small tolerances
geo = geodesic(yy, J, secondderiv, 3, 3, thetainit, vinit, atol = .1, rtol =  .1, callback = callback,invSVD=False, parameterspacenorm=True)  

# Integrate
geo.integrate(25,maxsteps=1000)

fig2 = plt.figure()
ax2 = fig2.add_subplot(111)
ax2.plot(geo.ts, geo.xs,label=('$\theta_0$','$\theta_1$','$\theta_2$'))
ax2.set_xlabel(r'$\tau$')
ax2.set_ylabel(r'$\theta(\tau)$')
plt.legend()
plt.show()
