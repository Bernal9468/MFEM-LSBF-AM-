# Residual stress for the single-layer full bridge (Enzyme/RESIDUAL only, no OTI)

Scope agreed with the user (2026-10-14):
- thermal history = the existing single-layer window (t = 1.2522 .. 1.2566, 4400 steps of 1 us, laser on the first patch),
  then the laser is switched off and the part cools to ambient / substrate temperature;
- material = existing Hill/J2 thermoplastic model (bridge_4layer_kokkos_OTI/mechanical/src/hill_*.hpp), checked against Abaqus
  on the same mesh;
- derivatives: Enzyme/RESIDUAL only (OTI is NOT needed for this part). Nominal first, then sensitivities.

## What exists / does not exist
- Thermal driver (opt/src): fixed dt only, laser never switches off, no field output. -> Phase 1 adds laser-off, growing dt, snapshots.
- Mechanical driver (bridge_4layer_kokkos_OTI/mechanical): ONE static solve on a frozen temperature field; plastic state stays 0.
  It has the pieces needed (J2/Hill return map, committed plastic state per qp, B-bar element) but no time loop.
- Abaqus reference deck (Abaqus_full_bridge_single_layer): heat transfer only; needs a sequentially coupled mechanical step.

## Phases
1. Thermal cool-down + snapshots (src here). Output: T_<step>.bin (float64, ndof) + snapshots.csv, nominal.
2. Incremental thermo-mechanical driver (nominal): loop over snapshots, eigenstrain alpha(T)(T-Tref), Newton equilibrium per
   snapshot, return map, CommitState; residual stress = state at the last snapshot. Elastic-only run first as a sanity check.
3. Abaqus mechanical deck on the same mesh (read temperatures from the thermal odb / snapshots), compare residual stress field.
4. Enzyme/RESIDUAL first-order sensitivities of the residual stress (thermal sensitivities already exist; mechanical part
   differentiated with Enzyme through the return map).

## Open modelling decisions (needed before Phase 2 results are meaningful)
- Unmelted powder elements of the bridge layer: stiffness/strength before melting, and strain reset on melting/solidification.
- Mechanical boundary conditions (substrate bottom fixed? symmetry?) and stress-free reference temperature.
- Temperature-dependent Young's modulus, yield, hardening, expansion for IN625 (take from the existing Abaqus-validated model).

## Final spec (user answers, 2026-10-14)
- Same single-layer mesh as the thermal Enzyme work (full_bridge_single_layer.mesh / inputs/full_bridge.mesh, 188,648 hexes, 229,919 nodes).
- Thermal side: NOMINAL ONLY (laser window + cool-down, snapshots). No thermal sensitivities.
- Mechanical side: nominal + first order + FULL second-order Hessian (pure and mixed) of the residual stress, Enzyme forward mode,
  with respect to 5 mechanical parameters (scale factors on the IN625 curves):  E, nu, CTE, yield stress sigma_y, hardening modulus H
  -> 5 first-order fields + 15 second-order terms (5 pure + 10 mixed).
- Material: J2Material of bridge_4layer_kokkos_OTI/mechanical/src/hill_thermoplastic_material.hpp (E(T), nu(T) tables to 1144 K,
  sigma_y 725 -> 990 MPa linear hardening over eps_p 0.35 so H = 757.1 MPa, Hill ratios R33 = 0.8483).
- Defaults chosen by me (not yet confirmed): substrate bottom fixed, stress-free reference = initial temperature, bridge elements
  stress-free until first melting (strain reference reset on solidification). Revisit when the Abaqus mechanical deck is built.

## Status
- Phase 1 code done: -laser-off, -cool-grow, -cool-dtmax, -snapdir, -snap-every (src/). Short test (40 heating steps + 20 s cool-down) OK.
- Next: nominal thermal run (window + 10 s cool-down) -> snapshots; then the mechanical driver.
