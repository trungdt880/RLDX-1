"""Phase 0.5c — pure-MuJoCo coupling / tracking check for the ALLEX humanoid.

OFFLINE sanity test (no robosuite, no policy). Drives ALLEX's 48 absolute
joint-position targets from a recorded motion through the MJCF's own `<position>`
actuators and checks:

  1. Tracking: achieved qpos vs commanded q, per-joint and per-group RMS/max error.
  2. Couplings: the 12 passively-coupled (`<equality>` joint2-side) joints follow
     their `polycoef` relation to the driver joint within tolerance.
  3. Stability: no NaN/inf in qpos/qvel; no joint wildly outside ctrlrange.

Drive scheme mirrors examples/mujoco/replay.py: native position-actuator PD +
gravity-comp feed-forward (qfrc_bias) + error-term D (kv * qd_ref via qfrc_applied);
per-frame kp/kv when the motion carries them. Ramps home->first frame (never
teleports qpos, which would snap the equality-coupled joints).

Run headless:
  MUJOCO_GL=egl python coupling_check.py --motion .../hello.npz
Prints PASS/FAIL summary; appends a per-joint error table to
coupling_check_report.txt.
"""
import argparse
from pathlib import Path

import numpy as np
import mujoco

HERE = Path(__file__).resolve().parent
DEFAULT_MODEL = Path.home() / "workspace" / "allex_model" / "mjcf" / "ALLEX.xml"
MOTIONS_DIR = Path.home() / "workspace" / "allex_model" / "examples" / "motions"
REPORT = HERE / "coupling_check_report.txt"

# Acceptance thresholds (degrees) on RMS tracking error, per group.
GROUP_RMS_TOL_DEG = {
    "left_arm": 5.0, "right_arm": 5.0,
    "left_hand": 12.0, "right_hand": 12.0,   # fingers lag more; report honestly
    "neck": 5.0, "waist": 5.0,
}
# Coupling residual tolerances. The waist couplings behave rigidly; the finger
# DIP / thumb IP followers are UN-actuated and held only by MuJoCo's default SOFT
# equality constraint (solref [0.02,1] = 20ms time constant), so under fast finger
# flexion they lag transiently. We therefore check (a) rigid waist within a tight
# bound, (b) finger couplings resolve on average (RMS) and never DIVERGE (bounded
# transient max), while reporting the transient max honestly.
WAIST_RES_TOL_DEG = 2.0        # max residual for waist couplings (effectively rigid)
FINGER_RES_RMS_TOL_DEG = 15.0  # RMS residual for soft finger couplings
FINGER_RES_MAX_TOL_DEG = 90.0  # transient max bound: above this = divergence/blow-up


def joint_group(name: str) -> str:
    arm = ("Shoulder", "Elbow", "Wrist")
    if name.startswith("Neck"):
        return "neck"
    if name.startswith("Waist"):
        return "waist"
    side = "left" if name.startswith("L_") else ("right" if name.startswith("R_") else None)
    if side is None:
        return "other"
    if any(k in name for k in arm):
        return f"{side}_arm"
    return f"{side}_hand"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=str(DEFAULT_MODEL))
    ap.add_argument("--motion", default=str(MOTIONS_DIR / "hello.npz"))
    args = ap.parse_args()

    motion_name = Path(args.motion).stem
    z = np.load(args.motion, allow_pickle=True)
    names = [str(x) for x in z["joint_names"]]
    q = z["q"].astype(np.float64)            # [T, 48] rad, absolute targets
    dt = float(z["dt"])
    qd = z["qd"].astype(np.float64)
    kp_seq = z["kp"].astype(np.float64) if "kp" in z else None
    kv_seq = z["kv"].astype(np.float64) if "kv" in z else None
    T = len(q)

    m = mujoco.MjModel.from_xml_path(args.model)
    m.opt.timestep = dt
    d = mujoco.MjData(m)
    sd = mujoco.MjData(m)                     # scratch for gravity comp (qvel=0)

    # driven joint -> (motion col, actuator id, dof addr, nominal kv)
    pairs, qadr = [], {}
    for a in range(m.nu):
        jid = int(m.actuator_trnid[a, 0])
        jn = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, jid)
        qadr[jn] = int(m.jnt_qposadr[jid])
        if jn in names:
            kv = float(-m.actuator_biasprm[a, 2])
            pairs.append((names.index(jn), a, int(m.jnt_dofadr[jid]), kv))
    # qpos addr for every joint (incl. coupled ones without actuators)
    for jid in range(m.njnt):
        jn = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, jid)
        qadr[jn] = int(m.jnt_qposadr[jid])

    # ctrlrange per actuator (for stability bound-check)
    ctrlrange = {names.index(mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, int(m.actuator_trnid[a, 0]))):
                 (float(m.actuator_ctrlrange[a, 0]), float(m.actuator_ctrlrange[a, 1]))
                 for a in range(m.nu)
                 if mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, int(m.actuator_trnid[a, 0])) in names}

    # equality jointpos couplings. MuJoCo mjEQ_JOINT semantics: joint1 is the
    # DEPENDENT (passive follower) joint, joint2 is the DRIVER. The enforced
    # relation is:  (pos1 - ref1) = poly(pos2 - ref2). The follower (joint1)
    # here is always the un-actuated coupled dof (waist upper/dummy, finger DIP,
    # thumb IP); the driver (joint2) is the actuated joint (waist lower, PIP, MCP).
    couplings = []   # (label, adr_dep, adr_drv, ref_dep, ref_drv, polycoef)
    for e in range(m.neq):
        if m.eq_type[e] != mujoco.mjtEq.mjEQ_JOINT:
            continue
        if not m.eq_active0[e]:
            continue
        j_dep = int(m.eq_obj1id[e]); j_drv = int(m.eq_obj2id[e])
        n_dep = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j_dep)
        n_drv = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j_drv)
        a_dep = int(m.jnt_qposadr[j_dep]); a_drv = int(m.jnt_qposadr[j_drv])
        couplings.append((f"{n_dep}<-{n_drv}", a_dep, a_drv, float(m.qpos0[a_dep]),
                          float(m.qpos0[a_drv]), m.eq_data[e, :5].copy()))

    def gravity_comp():
        sd.qpos[:] = d.qpos
        sd.qvel[:] = 0.0
        mujoco.mj_forward(m, sd)
        return sd.qfrc_bias

    def step_to(fr, frd, kpf, kvf):
        d.qfrc_applied[:] = gravity_comp()
        for c, a, dof, kv0 in pairs:
            if kpf is not None:
                kp = kpf[c]; kv = kvf[c]
                m.actuator_gainprm[a, 0] = kp
                m.actuator_biasprm[a, 1] = -kp
                m.actuator_biasprm[a, 2] = -kv
            else:
                kv = kv0
            d.ctrl[a] = fr[c]
            d.qfrc_applied[dof] += kv * frd[c]
        mujoco.mj_step(m, d)

    # ramp home -> first frame (no teleport)
    home = np.array([d.qpos[qadr[n]] for n in names])
    nwarm = max(1, int(round(0.5 / dt)))
    zero = np.zeros(len(names))
    kp0 = kp_seq[0] if kp_seq is not None else None
    kv0 = kv_seq[0] if kv_seq is not None else None
    for k in range(nwarm):
        alpha = (k + 1) / nwarm
        step_to(home * (1.0 - alpha) + q[0] * alpha, zero, kp0, kv0)

    achieved = np.zeros((T, len(names)))
    coup_res = np.zeros((T, len(couplings)))
    unstable = False
    ctrl_viol = []   # (name, achieved_deg, lo_deg, hi_deg)

    for i in range(T):
        step_to(q[i], qd[i],
                kp_seq[i] if kp_seq is not None else None,
                kv_seq[i] if kv_seq is not None else None)
        for j, n in enumerate(names):
            achieved[i, j] = d.qpos[qadr[n]]
        for k, (_, a_dep, a_drv, r_dep, r_drv, poly) in enumerate(couplings):
            x = d.qpos[a_drv] - r_drv
            pred = r_dep + poly[0] + poly[1]*x + poly[2]*x**2 + poly[3]*x**3 + poly[4]*x**4
            coup_res[i, k] = d.qpos[a_dep] - pred
        if not (np.all(np.isfinite(d.qpos)) and np.all(np.isfinite(d.qvel))):
            unstable = True
            print(f"[{motion_name}] NaN/inf detected at step {i}", flush=True)
            break

    # stability: joints far outside ctrlrange (allow 0.2 rad slack)
    last = achieved[-1]
    for c, a, dof, kv0 in pairs:
        n = names[c]
        lo, hi = ctrlrange.get(c, (-np.inf, np.inf))
        val = last[c]
        if val < lo - 0.2 or val > hi + 0.2:
            ctrl_viol.append((n, np.degrees(val), np.degrees(lo), np.degrees(hi)))

    # per-joint tracking error (deg)
    err = np.degrees(achieved - q)
    rms_j = np.sqrt(np.mean(err**2, axis=0))
    max_j = np.max(np.abs(err), axis=0)

    # per-group
    groups = {}
    for j, n in enumerate(names):
        groups.setdefault(joint_group(n), []).append(j)
    group_stats = {}
    for g, idxs in sorted(groups.items()):
        e = err[:, idxs]
        group_stats[g] = (float(np.sqrt(np.mean(e**2))), float(np.max(np.abs(e))))

    # coupling residuals (deg)
    coup_res_deg = np.degrees(coup_res)
    coup_rms = np.sqrt(np.mean(coup_res_deg**2, axis=0))
    coup_max = np.max(np.abs(coup_res_deg), axis=0)

    # ---- verdicts ----
    track_pass = all(group_stats[g][0] <= GROUP_RMS_TOL_DEG.get(g, 12.0)
                     for g in group_stats)
    is_waist = np.array(["Waist" in lbl for (lbl, *_ ) in couplings])
    waist_ok = bool(np.all(coup_max[is_waist] <= WAIST_RES_TOL_DEG)) if is_waist.any() else True
    finger_rms_ok = bool(np.all(coup_rms[~is_waist] <= FINGER_RES_RMS_TOL_DEG)) if (~is_waist).any() else True
    finger_bounded = bool(np.all(coup_max[~is_waist] <= FINGER_RES_MAX_TOL_DEG)) if (~is_waist).any() else True
    coup_pass = waist_ok and finger_rms_ok and finger_bounded
    stable = (not unstable) and (len(ctrl_viol) == 0)
    overall = track_pass and coup_pass and stable

    # ---- print summary ----
    print(f"\n===== ALLEX coupling/tracking check : {motion_name} "
          f"({T} frames, {T*dt:.2f}s, dt={dt*1000:.1f}ms) =====")
    print("  per-group tracking error (deg):")
    for g in sorted(group_stats):
        rms, mx = group_stats[g]
        tol = GROUP_RMS_TOL_DEG.get(g, 12.0)
        tag = "OK " if rms <= tol else "HI "
        print(f"    {tag} {g:11s}  RMS={rms:6.3f}  max={mx:7.3f}   (tol RMS<={tol})")
    w = coup_max[is_waist]; fr = coup_rms[~is_waist]; fm = coup_max[~is_waist]
    print(f"  coupling residuals (deg):")
    print(f"    waist  (rigid): max={w.max():.4f}  (tol<={WAIST_RES_TOL_DEG}) "
          f"-> {'OK' if waist_ok else 'FAIL'}")
    print(f"    finger (soft) : RMS max={fr.max():.3f} (tol RMS<={FINGER_RES_RMS_TOL_DEG}), "
          f"transient max={fm.max():.3f} (bound<={FINGER_RES_MAX_TOL_DEG}, no-divergence) "
          f"-> {'OK' if (finger_rms_ok and finger_bounded) else 'FAIL'}")
    if not stable:
        if unstable:
            print("  STABILITY: FAIL (NaN/inf)")
        for n, v, lo, hi in ctrl_viol:
            print(f"  STABILITY: {n} = {v:.1f} deg outside ctrlrange [{lo:.1f},{hi:.1f}]")
    else:
        print("  stability: OK (finite qpos/qvel, joints within ctrlrange)")
    print(f"  VERDICT [{motion_name}]: "
          f"{'PASS' if overall else 'FAIL'}  "
          f"(track={'P' if track_pass else 'F'} "
          f"coupling={'P' if coup_pass else 'F'} "
          f"stable={'P' if stable else 'F'})\n")

    # ---- write per-joint table ----
    with open(REPORT, "a") as f:
        f.write(f"\n===== {motion_name}  ({T} frames, {T*dt:.2f}s, dt={dt*1000:.1f}ms, "
                f"kp/kv={'per-frame' if kp_seq is not None else 'MJCF-nominal'}) =====\n")
        f.write("  per-joint tracking error (deg):\n")
        f.write(f"    {'joint':28s} {'group':11s} {'RMS':>8s} {'max':>8s}\n")
        order = np.argsort(-rms_j)
        for j in order:
            f.write(f"    {names[j]:28s} {joint_group(names[j]):11s} "
                    f"{rms_j[j]:8.3f} {max_j[j]:8.3f}\n")
        f.write("  per-group tracking error (deg):\n")
        for g in sorted(group_stats):
            rms, mx = group_stats[g]
            f.write(f"    {g:11s} RMS={rms:7.3f} max={mx:8.3f} "
                    f"tol={GROUP_RMS_TOL_DEG.get(g,12.0)} "
                    f"{'OK' if rms <= GROUP_RMS_TOL_DEG.get(g,12.0) else 'HI'}\n")
        f.write("  equality coupling residuals (deg)  [follower<-driver]:\n")
        for k, (lbl, *_ ) in enumerate(couplings):
            kind = "waist/rigid" if is_waist[k] else "finger/soft"
            f.write(f"    {lbl:42s} {kind:11s} RMS={coup_rms[k]:8.4f} max={coup_max[k]:8.4f}\n")
        f.write(f"  stability: {'OK' if stable else 'FAIL'}"
                f"{'' if stable else ' ' + str(ctrl_viol)}\n")
        f.write(f"  VERDICT: {'PASS' if overall else 'FAIL'}\n")

    return overall


if __name__ == "__main__":
    main()
