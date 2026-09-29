// Module 24F part 2: PDLP with the iterate RESIDENT on the GPU.
//
// Stage E put `K x` and `K' y` on the device and measured the honest result on
// datt256: kernel 1.63 s, host<->device copies 2.94 s. Nearly two-thirds of the
// GPU time was the bus, because `MatVec` takes host spans and every product had
// to ship its input down and its output back.
//
// This implementation of `pdlp::IterationBackend` keeps every hot-path vector
// on the device for the whole solve -- the iterate, the trial point, the
// extrapolation, the three products Algorithm 2 needs, the infeasibility
// sequences and the restart average. What crosses the bus:
//
//   per trial                 3 doubles (TrialMetrics), 24 bytes
//   per check_interval (40)   the vectors the cold path reads
//   once                      the matrix, both orientations, and c, b, l, u
//
// So the per-product transfer that Stage E measured is gone, and in its place
// is one 24-byte copy per trial -- which is also the only synchronization
// point in the hot loop. It cannot be removed: Algorithm 2's accept/reject is a
// decision the host makes on those three numbers.
//
// DETERMINISM. The three reductions are done in two stages with a FIXED grid
// (per-block partial sums, then a single block summing those in order), not
// with atomics. Atomic adds on doubles land in whatever order the hardware
// schedules, which would make two runs of the same model take different
// numbers of steps. A fixed grid makes the device path reproducible run to run.
// It does NOT make it bit-identical to the host path: the summation ORDER
// differs, and nvcc contracts `x - tau*(c - k)` into a fused multiply-add,
// which rounds once where the host rounds twice. Section 24E already measured
// what that does to an adaptive method -- the step counts drift by a handful --
// and the tests here compare against the host with a tolerance for that reason.
//
// THE COLD PATH still needs `K x` on host vectors, once per 40 iterations. It is
// served by `cold_matvec()`, which uses the SAME device copy of the matrix --
// so the model is resident once, not twice.
//
// Declared with a plain C++ interface and an opaque implementation so this
// header compiles with any C++ compiler; the default `release` preset includes
// it in header_compile_check with no CUDA toolkit present.

#ifndef SOVSOLVE_SOLVER_GPU_PDLP_DEVICE_HPP
#define SOVSOLVE_SOLVER_GPU_PDLP_DEVICE_HPP

#include <cstddef>
#include <memory>
#include <string>
#include <optional>

#include "sovsolve/core/Status.hpp"
#include "sovsolve/core/Types.hpp"
#include "sovsolve/model/Canonical.hpp"
#include "sovsolve/solver/pdlp/IterationBackend.hpp"
#include "sovsolve/solver/pdlp/MatVec.hpp"

namespace sovsolve::solver::gpu {

using core::Real;

class DevicePdlpBackend final : public pdlp::IterationBackend {
 public:
  /// Uploads the matrix (both orientations) and `c`, `b`, `l`, `u`, and
  /// allocates every hot-path vector. Fails rather than throwing if there is no
  /// device or an allocation fails.
  [[nodiscard]] static core::Expected<std::unique_ptr<DevicePdlpBackend>> create(
      const model::CanonicalProblem& problem);

  ~DevicePdlpBackend() override;
  DevicePdlpBackend(const DevicePdlpBackend&) = delete;
  DevicePdlpBackend& operator=(const DevicePdlpBackend&) = delete;

  void set_iterate(core::HostSpan<const Real> x, core::HostSpan<const Real> y) override;
  void download(pdlp::BackendVector which, core::HostSpan<Real> out) override;
  void begin_step() override;
  [[nodiscard]] pdlp::TrialMetrics trial(Real tau, Real sigma) override;
  void accept_trial() override;
  void fixed_step(Real tau, Real sigma) override;
  [[nodiscard]] std::optional<Real> spectral_norm(std::size_t iterations,
                                                  Real tolerance) override;
  [[nodiscard]] bool supports_halpern() const override { return true; }
  void begin_halpern() override;
  [[nodiscard]] pdlp::TrialMetrics halpern_step(Real eta, Real omega, Real gamma,
                                                Real lambda) override;
  [[nodiscard]] pdlp::AnchorDistance restart_at_pdhg_point() override;
  void write_halpern_state(const pdlp::HalpernState& state) override;
  [[nodiscard]] pdlp::HalpernState read_halpern_state() override;
  void run_halpern(std::size_t count, std::uint64_t first_iteration,
                   const pdlp::HalpernParams& params, bool track_differences) override;
  [[nodiscard]] bool evaluate_resident(core::HostSpan<const Real> row_scale,
                                       core::HostSpan<const Real> col_scale,
                                       ResidentSums& out) override;
  void snapshot_iterate() override;
  void finish_difference() override;
  void accumulate_average(Real weight) override;
  void reset_average() override;
  [[nodiscard]] core::Status status() const override;
  [[nodiscard]] std::size_t own_products() const override;

  /// CUDA graphs over the resident Halpern loop (`run_halpern`). On by
  /// default; off launches the identical kernel sequence directly, which is
  /// what makes the two comparable.
  void set_use_graphs(bool enabled);
  /// True once a graph has actually been captured and is in use.
  [[nodiscard]] bool graphs_active() const;
  /// Why capture was abandoned, if it was. Empty otherwise.
  [[nodiscard]] const std::string& graph_note() const;
  /// Graph launches issued (or, without graphs, iterations issued).
  [[nodiscard]] std::size_t graph_launches() const;

  /// The cold path's `K x` / `K' y`, over the same device matrix. Pass this as
  /// the `MatVec` argument of `solve_pdlp` alongside the backend itself.
  [[nodiscard]] pdlp::MatVec& cold_matvec();

  /// Bytes resident on the device: matrix, problem vectors, every hot-path
  /// vector, cuSPARSE workspaces and reduction scratch.
  [[nodiscard]] std::size_t device_bytes() const;

  /// Opaque; defined in PdlpDevice.cu. Named here, not hidden, only so the
  /// cold-path MatVec in that file can reach the same device state -- it is
  /// incomplete to every other translation unit.
  struct Impl;

 private:
  explicit DevicePdlpBackend(std::unique_ptr<Impl> impl);
  std::unique_ptr<Impl> impl_;
  std::unique_ptr<pdlp::MatVec> cold_;
};

}  // namespace sovsolve::solver::gpu

#endif  // SOVSOLVE_SOLVER_GPU_PDLP_DEVICE_HPP
