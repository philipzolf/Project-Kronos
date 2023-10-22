import numpy as np
import matplotlib.pyplot as plt

def rk4(u_k, delta_t):
    k1 = delta_t*f_true(u_k)
    k2 = delta_t*f_true(u_k+k1/2)
    k3 = delta_t*f_true(u_k+k2/2)
    k4 = delta_t*f_true(u_k+k3)

    ukplus1 = u_k + (k1 + 2*k2 + 2*k3 + k4)/6
    return ukplus1

def ivp_rk4(u_0, T, delta_t):
    times = []
    u = []
    num_elements = int(T / delta_t) + 1
    for t_k in np.linspace(0,T,num_elements):
        times.append(t_k)
        u.append(u_0)
        u_k = rk4(u_0, delta_t)
        u_0 = u_k
    return np.array(u), times

G = 6.674e-11
def f_true(u):
    r = u[0]
    v = u[1]
    N = r.shape[0]
    rddot = np.zeros((N,3))
    for i in range(0,N):
        for j in range(0,N):
            if i == j:
                continue
            rddot[i] += G*m[j]*(r[j]-r[i])/(np.linalg.norm(r[j]-r[i])**3)
    udot = np.zeros(u.shape)
    udot[0] = v
    udot[1] = rddot
    return udot

def ab4(u_0,T,delta_t):
    k = int(T/delta_t) + 1
    times = np.linspace(0, T, k)
    u = []
    u.append(u_0)
    u.append(rk4(u[0],delta_t))
    u.append(rk4(u[1],delta_t))
    u.append(rk4(u[2],delta_t))
    for i in range (4,k):
        u.append(u[i-1] + (delta_t/24) * (55*f_true(u[i-1]) - 59*f_true(u[i-2]) + 37*f_true(u[i-3]) - 9*f_true(u[i-4])))
    return np.array(u),times

# T, dt = 300 * 365 * 24 * 3600, 60*60*24*100
T, dt = 24*60*60*365*1, 24*60*60 
mSaturn = 568.32e24
mTitan = 1.345e23
rTitan = -1.22e9
mDeathStar = 2.24e23
# mearth = 5.97e24
mMimas = 3.75e19
rMimas = 198e6
mHyperion = 5.58*1080 # kg
rHyperion = 1.5e9 # m
m = np.array([mSaturn,mTitan,mDeathStar,mHyperion])
r0 = np.zeros((4,3))
r0[0] = [0.0, 0.0, 0.0]
r0[1] = [rTitan, 0.0,0.0]
r0[2] = [1.22e9*2, 0.0, 0.0]
r0[3] = [rHyperion, 0.0, 0.0]
# print(r0)
v0 = np.zeros((4,3))
v0[0] = [0.0, 0.0, 0.0]
v0[1] = [0.0, -np.sqrt(G*m[0]/np.linalg.norm(r0[1])), 0.0]
v0[2] = [0.0, -np.sqrt(G*m[0]/np.linalg.norm(r0[2])), 0.0]
v0[3] = [0.0, -np.sqrt(G*m[0]/np.linalg.norm(r0[3])), 0.0]

u0 = np.array([r0, v0])
u_rk, times = ivp_rk4(u0, T, dt)

fig = plt.figure()
ax = fig.add_subplot(projection='3d')

ur = u_rk[:,0]
uv = u_rk[:,1]

r1 = []
r2 = []
r3 = []
r4 = []
for i in range(ur.shape[0]):
    r1.append(ur[i][0])
    r2.append(ur[i][1])
    r3.append(ur[i][2])
    r4.append(ur[i][3])
r1 = np.array(r1)
r2 = np.array(r2)
r3 = np.array(r3)
r4 = np.array(r4)

ax.scatter(r1[:,0], r1[:,1], r1[:,2], color="yellow", label="Saturn")
ax.scatter(r2[:,0], r2[:,1], r2[:,2], color="blue", label="Titan")
ax.scatter(r3[:,0], r3[:,1], r3[:,2], color="black", label="Death Star")
ax.scatter(r4[:,0], r4[:,1], r4[:,2], color="cyan", label="Hyperion")
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
fig.suptitle("N Body Dynamics")
# plt.xlim(-1.22e12,1.22e9)
# plt.ylim(0, 1.22e9)
plt.legend()
plt.show()