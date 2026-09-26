// Module 31: the cuPDLPx restart and primal-weight controller, ONCE.
//
// This is the part of the reflected-Halpern scheme that makes DECISIONS: when
// to restart, and how to move the primal weight when it does. It used to live
// in Pdlp.cpp, run on the host, and read three scalars back from the backend
// after every single iteration to make them. On the device-resident path that
// read was a synchronization per iteration -- the host could never queue more
// than one iteration's kernels ahead of the GPU -- and on maros-r7 it made the
// GPU slower than the host outright.
//
// So the decisions move to wherever the iterate lives. A device backend runs
// them in a one-thread kernel, reading the scalars from device memory, and the
// host only looks at the controller's state at the termination check, once
// every `check_interval` iterations -- which is where cuPDLPx makes them too.
//
// WHY A SHARED HEADER AND NOT A SECOND COPY. The functions below are
// `__host__ __device__` under nvcc and plain inline functions everywhere else,
// so the host backend and the device kernel execute the SAME source. A second
// copy in the .cu file would be two implementations of one algorithm that could
// drift apart -- the thing this project has refused everywhere else, and which
// here would be invisible: a restart firing one iteration late changes the
// iteration count, not the answer.
//
// Everything is plain data. No `std::` algorithms or limits, because device
// code cannot call them; `HUGE_VAL` and the global math functions are
// available to both compilers.
//
// Sources: arXiv 2507.14051 section 3 (the three conditions and the PID);
// arXiv 2407.16144 equation (10) and Algorithm 2 (the restart rule's theory,
// and why the new epoch starts at `T(z)`). The derivations and the measured
// constants are in solver/pdlp/Pdlp.hpp and model/Options.hpp.

#ifndef SOVSOLVE_SOLVER_PDLP_HALPERN_CONTROL_HPP
#define SOVSOLVE_SOLVER_PDLP_HALPERN_CONTROL_HPP

#include <cmath>
#include <cstddef>
#include <cstdint>

#include "sovsolve/core/Types.hpp"

#if defined(__CUDACC__)
#define SOVSOLVE_HOST_DEVICE __host__ __device__
#else
#define SOVSOLVE_HOST_DEVICE
#endif

namespace sovsolve::solver::pdlp {

using core::Real;

/// Everything the controller reads and never writes. Passed by value into a
/// kernel, so it must stay trivially copyable.
struct HalpernParams {
  Real eta = 0.0;           ///< the constant step, `0.998 / ||A||_2`
  Real gamma = 1.0;         ///< reflection, already clamped to [0, 1]
  Real sufficient = 0.2;    ///< condition (i),   HPR-LP alpha_1
  Real necessary = 0.6;     ///< condition (ii),  HPR-LP alpha_2
  Real artificial = 0.2;    ///< condition (iii), HPR-LP alpha_3
  Real kp = 0.3;
  Real ki = 0.01;
  Real kd = 0.05;
  Real integral_clamp = 10.0;
  /// `||b||` and `||c||` of the problem the iteration runs on, for the
  /// relative infeasibilities in HPR-LP's safeguard (18). Constants of the
  /// run, so they travel in the parameters.
  Real b_norm = 0.0;
  Real c_norm = 0.0;
  std::uint64_t check_interval = 40;  ///< condition (iii)'s schedule
  /// Conditions (i) and (ii) are tested every `restart_check_every`
  /// iterations. 1 is cuPDLPx ("evaluates potential restart conditions at
  /// each iteration"); HPR-LP checks "every 150 iterations", and its
  /// published alpha constants were chosen under that schedule.
  std::uint64_t restart_check_every = 1;
  std::int32_t restarts_enabled = 1;
  std::int32_t weight_update = 1;
  /// 0: HPR-LP's rule (the default). 1: cuPDLPx's PID. See halpern_on_restart.
  std::int32_t weight_rule = 0;
};

/// Everything the controller writes. Lives with the iterate -- on the device
/// for a device backend -- and is read back only at a termination check.
struct HalpernState {
  Real omega = 1.0;          ///< primal weight; cuPDLPx starts it at 1
  Real reference = HUGE_VAL; ///< `r(z^{n,0})`, set by the epoch's first step
  Real last = HUGE_VAL;      ///< `r(z^{n,k-1})`, for condition (ii)
  Real pid_integral = 0.0;
  Real pid_last_error = 0.0;
  std::uint64_t inner = 0;     ///< `k`, iterations into the current epoch
  std::uint64_t restarts = 0;  ///< `n`
  /// Global iteration count, advanced by whoever runs the controller. Held
  /// HERE rather than passed in per call so that a device iteration's launch
  /// sequence is identical from one iteration to the next -- the property a
  /// CUDA graph needs, since a captured launch replays its arguments verbatim.
  std::uint64_t total = 0;
  std::int32_t have_pid_error = 0;
  /// Set by `halpern_observe`, read by whatever performs the restart. A flag
  /// rather than a return value so a device backend can guard kernels on it
  /// without the host ever seeing it.
  std::int32_t restart_pending = 0;
};

/// `x - x == 0` holds exactly for finite `x` (NaN and infinity both give NaN).
/// Spelled out because `std::isfinite` is not callable from device code and
/// the global `isfinite` is not reliably declared by <cmath> on every host.
SOVSOLVE_HOST_DEVICE inline bool halpern_finite(Real v) { return v - v == 0.0; }

/// One row's contribution to HPR-LP's primal infeasibility `Pi_D(b - A x)`:
/// an equality row is violated either way, a `<=` row only upward.
SOVSOLVE_HOST_DEVICE inline Real halpern_primal_violation(Real ax, Real b, bool equality) {
  const Real slack = ax - b;
  return equality ? slack : (slack > 0.0 ? slack : 0.0);
}

/// One column's contribution to HPR-LP's dual infeasibility `c - A'y - z`:
/// the part of the reduced cost `raw = c - A'y` that no bound can absorb.
/// The projection is Pdlp.cpp's `project_reduced_cost`, restated with plain
/// comparisons against the `core::INF` sentinel so device code can run it.
SOVSOLVE_HOST_DEVICE inline Real halpern_dual_leftover(Real raw, Real lower, Real upper) {
  const bool has_lower = lower > -core::INF;
  const bool has_upper = upper < core::INF;
  Real absorbed = raw;
  if (!has_lower && !has_upper) {
    absorbed = 0.0;
  } else if (!has_lower) {
    absorbed = raw < 0.0 ? raw : 0.0;
  } else if (!has_upper) {
    absorbed = raw > 0.0 ? raw : 0.0;
  }
  return raw - absorbed;
}

/// `lambda_k = (k+1)/(k+2)`, with `k` counted within the epoch -- the reset
/// at every restart is what re-weights a fresh anchor back up to 1/2.
SOVSOLVE_HOST_DEVICE inline Real halpern_lambda(const HalpernState& s) {
  const Real k = static_cast<Real>(s.inner);
  return (k + 1.0) / (k + 2.0);
}

/// `r(z) = ||z - PDHG(z)||_P` from the three scalars a step produced:
///
///     r^2 = (omega/eta) ||dx||^2 + 2 dy' A dx + ||dy||^2 / (eta omega)
///
/// `P` is positive definite for `eta < 1/||A||_2`, so `r^2 >= 0` in exact
/// arithmetic; the guard is for rounding where the terms nearly cancel.
SOVSOLVE_HOST_DEVICE inline Real halpern_fixed_point_error(Real eta, Real omega,
                                                           Real interaction, Real dx_sq,
                                                           Real dy_sq) {
  const Real r_sq = (omega / eta) * dx_sq + 2.0 * interaction + dy_sq / (eta * omega);
  return r_sq > 0.0 ? ::sqrt(r_sq) : 0.0;
}

/// Records `r(z^{n,k})` and decides whether to restart. `total` is the
/// iteration count AFTER this step.
///
/// The three conditions (cuPDLPx section 3):
///
///   (i)   r <= sufficient * r(z^{n,0})                        decisive decay
///   (ii)  r <= necessary  * r(z^{n,0})  and  r > r(z^{n,k-1}) decay, stalled
///   (iii) k >= artificial * total                             epoch too long
///
/// (i) and (ii) read data and are free, so they are tested every iteration.
/// (iii) reads only two counters, and tested every iteration it is
/// degenerate: at `total = 1` the epoch is 1 of 1 and `1 >= 0.36` fires, so a
/// run opens with a restart on nearly every step, each feeding the PID a
/// one-step distance. That stalled plain Halpern on a three-variable LP. It is
/// tested on the `check_interval` schedule averaged PDLP always used.
SOVSOLVE_HOST_DEVICE inline bool halpern_observe(const HalpernParams& p, HalpernState& s,
                                                 Real r, std::uint64_t total) {
  if (s.inner == 0) s.reference = r;
  ++s.inner;
  bool restart = false;
  // On HPR-LP's schedule (i) and (ii) are only EVALUATED every
  // `restart_check_every` iterations, and "no local progress" then compares
  // against the previous evaluation, not the previous iteration -- so `last`
  // only moves on a checked iteration too.
  const bool checked = p.restart_check_every <= 1 || total % p.restart_check_every == 0;
  if (!checked) {
    s.restart_pending = 0;
    // (iii) still runs on its own schedule below; fall through only for it.
    if (p.restarts_enabled != 0 && p.check_interval != 0 &&
        total % p.check_interval == 0 &&
        static_cast<Real>(s.inner) >= p.artificial * static_cast<Real>(total)) {
      s.restart_pending = 1;
      return true;
    }
    return false;
  }
  if (p.restarts_enabled != 0) {
    if (r <= p.sufficient * s.reference) {
      restart = true;
    } else if (r <= p.necessary * s.reference && r > s.last) {
      restart = true;
    } else if (p.check_interval != 0 && total % p.check_interval == 0 &&
               static_cast<Real>(s.inner) >= p.artificial * static_cast<Real>(total)) {
      restart = true;
    }
  }
  if (!restart) s.last = r;
  s.restart_pending = restart ? 1 : 0;
  return restart;
}

/// The controller's half of a restart, given how far the new anchor `T(z)`
/// sits from the old one. The iterate's half -- moving to `T(z)` -- belongs to
/// the backend.
///
/// The PID (cuPDLPx section 3) on `e_n = log(w ||dx|| / ||dy||)`:
///
///     log w_{n+1} = log w_n - [K_P e_n + K_I sum e_i + K_D (e_n - e_{n-1})]
///
/// With `K_I = K_D = 0` this is cuPDLP's Algorithm 3 with `theta = K_P`
/// exactly; the derivation is in Pdlp.hpp. The integral is clamped against
/// windup and the weight to `e^{+-30}`, a guard far outside any real balance
/// and far inside where either step size would denormalize.
SOVSOLVE_HOST_DEVICE inline void halpern_on_restart(const HalpernParams& p,
                                                    HalpernState& s, Real dx, Real dy,
                                                    Real primal_error, Real dual_error) {
  if (p.weight_update != 0 && p.weight_rule == 0) {
    // HPR-LP (arXiv 2408.12179) Algorithm 3, "SigmaUpdate", translated to
    // this method's primal weight. arXiv 2509.23903 Proposition 3.1 shows
    // cuPDLPx with gamma = 1 generates EXACTLY the HPR iterates when
    // `sigma = eta / omega` and `lambda_A = 1 / eta^2`; HPR-LP's (22) is
    // `sigma+ = (1/sqrt(lambda_A)) ||dx|| / ||dy|| = eta ||dx|| / ||dy||`,
    // hence `omega+ = eta / sigma+ = ||dy|| / ||dx||` -- Algorithm 3 of the
    // original PDLP paper with theta = 1, no smoothing, no free constant.
    //
    // Applied only under HPR-LP's two safeguards, else RESET:
    //   (17) dx, dy in (1e-16, 1e12)            -- the estimates are usable
    //   (18) dual_error / primal_error in (1e-8, 1e8)
    //                                           -- neither side has run away
    // HPR-LP resets `sigma` to 1, its own starting value. Its units include
    // an extra normalization of b and c this solver does not do, so the
    // reset here is to THIS method's starting value, cuPDLPx's `omega = 1`.
    const bool usable = dx > 1e-16 && dx < 1e12 && dy > 1e-16 && dy < 1e12;
    const Real ratio = dual_error / primal_error;  // NaN or inf fails the test
    const bool balanced = ratio > 1e-8 && ratio < 1e8;
    s.omega = (usable && balanced) ? dy / dx : 1.0;
  }
  // The PID below is cuPDLPx's rule, kept as the alternative (weight_rule 1).
  // An anchor that did not move in one block says nothing about the balance.
  constexpr Real kTiny = 1e-12;
  if (p.weight_update != 0 && p.weight_rule == 1 && dx > kTiny && dy > kTiny) {
    const Real error = ::log(s.omega * dx / dy);
    if (halpern_finite(error)) {
      const Real clamp = ::fabs(p.integral_clamp);
      Real integral = s.pid_integral + error;
      if (integral > clamp) integral = clamp;
      if (integral < -clamp) integral = -clamp;
      s.pid_integral = integral;
      const Real derivative = s.have_pid_error != 0 ? error - s.pid_last_error : 0.0;
      Real next = ::log(s.omega) - (p.kp * error + p.ki * integral + p.kd * derivative);
      if (halpern_finite(next)) {
        constexpr Real kLogWeightLimit = 30.0;
        if (next > kLogWeightLimit) next = kLogWeightLimit;
        if (next < -kLogWeightLimit) next = -kLogWeightLimit;
        s.omega = ::exp(next);
      }
      s.pid_last_error = error;
      s.have_pid_error = 1;
    }
  }
  s.inner = 0;
  s.reference = HUGE_VAL;
  s.last = HUGE_VAL;
  // `restart_pending` is deliberately NOT cleared here. On the device the
  // kernel that moves the iterate to `T(z)` runs AFTER this one and is
  // guarded on the flag; clearing it here would silently skip every restart's
  // move while still counting it. The next `halpern_observe` overwrites it.
  ++s.restarts;
}

}  // namespace sovsolve::solver::pdlp

#endif  // SOVSOLVE_SOLVER_PDLP_HALPERN_CONTROL_HPP
