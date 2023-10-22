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
m = np.array([568.32e24,1.345e23,2.24e23])
def f_true(u):
    r = u[0]
    v = u[1]
    rddot = np.zeros((3,3))
    for i in range(0,3):
        for j in range(0,3):
            if i == j:
                continue
            rddot[i] += G*m[j]*(r[j] - r[i])/np.linalg.norm(r[j] - r[i])**3
    return np.array([v, rddot])

T, dt = 300 * 365 * 24 * 3600, 60*60*24
# T, dt = 2000, 0.5 
r0 = np.zeros((3,3))
r0[0] = [0.0, 0.0, 0.0]
r0[1] = [-1.22e9, 0.0,0.0]
r0[2] = [1.22e9 + 1e7, 0.0, 0.0]
# print(r0)
v0 = np.zeros((3,3))
v0[0] = [0.0, 0.0, 0.0]
v0[1] = [0.0, -np.sqrt(G*m[1]/np.linalg.norm(r0[1])), 0.0]
v0[2] = [0.0, np.sqrt(G*m[2]/np.linalg.norm(r0[2])), 0.0]

u0 = np.array([r0, v0])
u_rk, times = ivp_rk4(u0, T, dt)

fig = plt.figure()
ax = fig.add_subplot(projection='3d')

ur = u_rk[:,0]
uv = u_rk[:,1]

rs = []
rt = []
rds = []
for i in range(ur.shape[0]):
    rs.append(ur[i][0])
    rt.append(ur[i][1])
    rds.append(ur[i][2])
rs = np.array(rs)
rt = np.array(rt)
rds = np.array(rds)

ax.scatter(rs[:,0], rs[:,1], rs[:,2], color="yellow")
ax.scatter(rt[:,0], rt[:,1], rt[:,2], color="blue")
ax.scatter(rds[:,0], rds[:,1], rds[:,2], color="green")
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
fig.suptitle("3 Body Dynamics")
# plt.xlim(-1.22e12,1.22e9)
# plt.ylim(0, 1.22e9)
plt.show()