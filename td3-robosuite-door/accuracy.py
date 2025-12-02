import numpy as np

# ----- robot params (edit q, link lengths if needed) -----
l2, l3, l4 = 0.2910, 0.3131, 0.2112
q = np.radians([70.0, 90.0, -20.0, 90.0, 70.0])   # t1..t5 in radians
# ---------------------------------------------------------

# # encoder counts and gear (A2 for joints 0..2, ST3020 for 3..4)
# counts = np.array([524288.0, 524288.0, 524288.0, 4096.0, 4096.0])
# gear   = np.array([9.0, 9.0, 18.0, 1.0, 1.0])    # motor_rev per joint_rev (9:1 for A2)


# encoder counts and gear (A2 for joints 0..2, ST3020 for 3..4)
counts = np.array([524288.0, 524288.0, 524288.0, 524288.0, 524288.0])
gear   = np.array([9.0, 9.0, 9.0, 9.0, 9.0])    # motor_rev per joint_rev (9:1 for A2)



# smallest joint step (radians) at joint output
joint_step = 2.0 * np.pi / (counts * gear)   # safe for both actuator types

# --- build Jacobian (same structure you provided) ---
t1, t2, t3, t4, t5 = q
t23 = t2 + t3
t234 = t23 + t4

c1, s1 = np.cos(t1), np.sin(t1)
c2, s2 = np.cos(t2), np.sin(t2)
c23, s23 = np.cos(t23), np.sin(t23)
c234, s234 = np.cos(t234), np.sin(t234)

J_pos = np.array([
    [-(l2*c2 + l3*c23 + l4*c234)*s1,
     -(l2*s2 + l3*s23 + l4*s234)*c1,
     -(l3*s23 + l4*s234)*c1,
     -l4*s234*c1,
     0.0],
    [(l2*c2 + l3*c23 + l4*c234)*c1,
     -(l2*s2 + l3*s23 + l4*s234)*s1,
     -(l3*s23 + l4*s234)*s1,
     -l4*s234*s1,
     0.0],
    [0.0,
     l2*c2 + l3*c23 + l4*c234,
     l3*c23 + l4*c234,
     l4*c234,
     0.0]
])
J_orient = np.array([
    [0.0, 1.0, 1.0, 1.0, 0.0],
    [0.0, 0.0, 0.0, 0.0, 1.0]
])
J = np.vstack((J_pos, J_orient))   # 5x5

# --- minimal end-effector change for one encoder tick per joint ---
dx_each = J @ joint_step            # 5-vector: [dx,dy,dz, dphi,dpsi]
pos_res = np.linalg.norm(dx_each[:3])
orient_res = np.linalg.norm(dx_each[3:])

# --- per-joint contributions (one tick on that joint only) ---
per_joint_pos_mm = np.zeros(5)
per_joint_orient_deg = np.zeros(5)
for i in range(5):
    dq = np.zeros(5); dq[i] = joint_step[i]
    dp = J_pos @ dq
    dor = J_orient @ dq
    per_joint_pos_mm[i] = np.linalg.norm(dp) * 1000.0
    per_joint_orient_deg[i] = np.degrees(np.linalg.norm(dor))

# --- prints ---
np.set_printoptions(precision=6, suppress=True)
print("Joint step (deg):", np.degrees(joint_step))
print("EE pos resolution for simultaneous single ticks (mm):", pos_res * 1000.0)
print("EE orient resolution for simultaneous single ticks (deg):", np.degrees(orient_res))
print("Per-joint EE pos (one tick) [mm]:", per_joint_pos_mm)
print("Per-joint EE orient (one tick) [deg]:", per_joint_orient_deg)

# Manufacturer accuracy note
print("\nNote: QDD A2 lists OUTPUT POSITION ACCURACY ±0.015° (after reduction).")
print("That is the real-world accuracy guarantee and is much larger than the per-count resolution.")
