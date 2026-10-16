"""Finite-difference definitions of the 20 wired thermal parameters (opt/src driver, -dorder 1 output).

Each entry maps a signed step to the driver command-line arguments of the perturbed nominal run, and gives the
step in the units of the AD derivative, so that   FD = (T(+h) - T(-h)) / (2 * unit_step(h)).

The AD derivative of parameter p is w.r.t. its RELATIVE scale s_p (p -> p (1 + s_p)), except X0 [per mm of
rigid toolpath shift] and TL [per native time unit of lead-time delay]. Nominal values below are the driver
defaults (full_bridge_activation_tier1_oti.cpp main(); C2K = 273, so T_ambient = 313 K and T_sub_bc = 353 K).
"""

KP_FRACTION, RHOP_FRACTION, HP_OVER_HC = 0.2, 0.6, 0.1
H_CONV = 0.018
T_AMB_C, T_SUB_C, C2K = 40.0, 80.0, 273.0
T_SOLIDUS, T_LIQUIDUS = 1563.0, 1623.0
GOLDAK = 0.1                 # -gaf/-gar/-gb/-gc of the production flags
X0_SCALE = 0.4               # X0 step [mm] per unit h  (h = 5e-3 -> 0.002 mm)
TL_POS_SCALE = 0.4           # TL step chosen so the laser moves TL_POS_SCALE*h mm (needs the scan speed)


def f(x):
    return repr(float(x))


def params(scan_speed):
    """name -> (sens column, args(v) for signed relative step v, unit_step(h), note)."""
    tl_scale = TL_POS_SCALE / scan_speed if scan_speed > 0 else 1e-6
    rel = lambda h: h
    P = {
        "k":    ("dT_dk_row1",    lambda v: f"-kscale {f(1 + v)}", rel, ""),
        "cp":   ("dT_dcp_row1",   lambda v: f"-cpscale {f(1 + v)}", rel, ""),
        "rho":  ("dT_drho_row1",  lambda v: f"-rhoscale {f(1 + v)}", rel, "bulk density only (rho_p has its own scale)"),
        "Lfus": ("dT_dLfus_row1", lambda v: f"-lfusscale {f(1 + v)}", rel, "zero until the probe melts"),
        "Ts":   ("dT_dTs_row1",   lambda v: f"-tsolidus {f(T_SOLIDUS * (1 + v))}", rel,
                 "FD also moves the powder->bulk switch threshold (discrete; AD holds it fixed)"),
        "Tliq": ("dT_dTliq_row1", lambda v: f"-tliquidus {f(T_LIQUIDUS * (1 + v))}", rel, ""),
        # hp = HP_OVER_HC * h_conv in the driver: perturbing -hconv alone would also scale hp (FD = dT/dhc + dT/dhp)
        "hc":   ("dT_dhc_row1",   lambda v: f"-hconv {f(H_CONV * (1 + v))} -hpratio {f(HP_OVER_HC / (1 + v))}", rel,
                 "-hpratio compensated so hp stays fixed"),
        "T0":   ("dT_dT0_row1",   lambda v: f"-tambc {f(T_AMB_C + (T_AMB_C + C2K) * v)}", rel, "ambient + initial temperature"),
        "P":    ("dT_dP_row1",    lambda v: f"-pscale {f(1 + v)}", rel, ""),
        "Eta":  ("dT_dEta_row1",  lambda v: f"-etascale {f(1 + v)}", rel, ""),
        "Rx":   ("dT_dRx_row1",   lambda v: f"-gaf {f(GOLDAK * (1 + v))} -gar {f(GOLDAK * (1 + v))}", rel, "front+rear semi-axes"),
        "Ry":   ("dT_dRy_row1",   lambda v: f"-gb {f(GOLDAK * (1 + v))}", rel, ""),
        "Rz":   ("dT_dRz_row1",   lambda v: f"-gc {f(GOLDAK * (1 + v))}", rel, ""),
        "X0":   ("dT_dX0_row1",   lambda v: f"-xshift {f(X0_SCALE * v)}", lambda h: X0_SCALE * h, "per mm"),
        "Vx":   ("dT_dVx_row1",   lambda v: f"-vxscale {f(v)}", rel, "time scaling about -ts; power schedule not shifted"),
        "TL":   ("dT_dTL_row1",   lambda v: f"-tshift {f(tl_scale * v)}", lambda h: tl_scale * h,
                 "per time unit; power schedule not shifted (same definition as AD)"),
        "Tw":   ("dT_dTw_row1",   lambda v: f"-tsub {f(T_SUB_C + (T_SUB_C + C2K) * v)}", rel,
                 "KNOWN OPEN ISSUE: 13-34% vs FD in README_IMPLEMENTATION.md"),
        "hp":   ("dT_dhp_row1",   lambda v: f"-hpratio {f(HP_OVER_HC * (1 + v))}", rel, ""),
        "kp":   ("dT_dkp_row1",   lambda v: f"-kpfrac {f(KP_FRACTION * (1 + v))}", rel, ""),
        "Rhop": ("dT_dRhop_row1", lambda v: f"-rhopfrac {f(RHOP_FRACTION * (1 + v))}", rel, ""),
    }
    return P


def scan_speed(laser_file, t0, t1):
    """Median speed [mm per native time unit] of the powered toolpath segments overlapping [t0, t1]."""
    import math
    rows = []
    with open(laser_file) as fh:
        for line in fh:
            s = line.replace(" ", "").split(",")
            if len(s) < 5:
                continue
            try:
                rows.append([float(x) for x in s[:5]])
            except ValueError:
                continue
    sp = []
    for a, b in zip(rows, rows[1:]):
        if b[0] <= a[0] or b[0] < t0 or a[0] > t1 or a[4] <= 0:
            continue
        d = math.dist(a[1:4], b[1:4])
        if d > 0:
            sp.append(d / (b[0] - a[0]))
    sp.sort()
    return sp[len(sp) // 2] if sp else 0.0
