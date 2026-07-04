import math

def vtrace(rewards, values, blp, tlp, terminated, bootstrap, gamma, rho_bar, c_bar):
    T = len(rewards)
    vt = [0.0]*T
    adv = [0.0]*T
    for t in reversed(range(T)):
        terminal = terminated[t]
        # Final-step precedence matches compute_gae_single_env: the last
        # rollout row bootstraps from `bootstrap` even if terminated[t] is
        # set. Interior terminal flags zero the bootstrap.
        if t == T-1:
            next_value = bootstrap
            next_vtrace = bootstrap
            next_v_minus_baseline = 0.0  # v_T - V_T == 0
        elif terminal:
            next_value = 0.0
            next_vtrace = 0.0
            next_v_minus_baseline = 0.0
        else:
            next_value = values[t+1]
            next_vtrace = vt[t+1]
            next_v_minus_baseline = vt[t+1] - values[t+1]
        ratio = math.exp(tlp[t] - blp[t])
        rho = min(rho_bar, ratio)
        c = min(c_bar, ratio)
        delta = rho * (rewards[t] + gamma*next_value - values[t])
        d = delta + gamma * c * next_v_minus_baseline
        vt[t] = values[t] + d
        adv[t] = rho * (rewards[t] + gamma*next_vtrace - values[t])
    return vt, adv, [min(rho_bar, math.exp(tlp[t]-blp[t])) for t in range(T)], [min(c_bar, math.exp(tlp[t]-blp[t])) for t in range(T)]

def fmt(xs):
    return "[" + ", ".join(f"{x:.10f}" for x in xs) + "]"

# ---- Scenario B: main reference, rho_bar=1, c_bar=1 (clipped rho at steps 0,2) ----
rewards = [1.0, 0.0, -1.0, 2.0]
values  = [0.5, 0.6, 0.7, 0.8]
blp     = [-0.5, -1.0, -0.7, -0.2]
tlp     = [-0.2, -1.5, -0.4, -0.6]
term    = [False, False, False, False]
boot    = 0.9
gamma   = 0.99
vt, adv, rho, c = vtrace(rewards, values, blp, tlp, term, boot, gamma, 1.0, 1.0)
print("Scenario B (rho_bar=1.0, c_bar=1.0):")
print("  ratios =", fmt([math.exp(tlp[i]-blp[i]) for i in range(4)]))
print("  rho    =", fmt(rho))
print("  c      =", fmt(c))
print("  vtrace_targets =", fmt(vt))
print("  advantages     =", fmt(adv))

# ---- Scenario C: independence rho_bar=0.5, c_bar=1.5 (same trajectory) ----
vt, adv, rho, c = vtrace(rewards, values, blp, tlp, term, boot, gamma, 0.5, 1.5)
print("\nScenario C (rho_bar=0.5, c_bar=1.5):")
print("  rho    =", fmt(rho))
print("  c      =", fmt(c))
print("  vtrace_targets =", fmt(vt))
print("  advantages     =", fmt(adv))

# ---- Scenario D: episode boundary, terminated at step 1 of 3-step ----
rewards = [1.0, 0.5, -0.5]
values  = [0.4, 0.6, 0.8]
blp     = [-0.5, -1.0, -0.7]
tlp     = [-0.2, -1.5, -0.4]
term    = [False, True, False]
boot    = 0.9
gamma   = 0.99
vt, adv, rho, c = vtrace(rewards, values, blp, tlp, term, boot, gamma, 1.0, 1.0)
print("\nScenario D (terminated at step 1, rho_bar=1.0, c_bar=1.0):")
print("  rho    =", fmt(rho))
print("  c      =", fmt(c))
print("  vtrace_targets =", fmt(vt))
print("  advantages     =", fmt(adv))
# Sanity: step0 must NOT carry signal from step2 (terminated at 1 blocks it)
print("  vtrace[0] with step2 reward changed to +100:")
vt2, _, _, _ = vtrace([1.0,0.5,100.0], values, blp, tlp, term, boot, gamma, 1.0, 1.0)
print("    vtrace_targets =", fmt(vt2), "(vtrace[0],vtrace[1] must be unchanged)")
