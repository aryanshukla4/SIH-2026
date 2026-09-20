// Cooperative cancellation for engines that race each other.
//
// The concurrent optimizer (solver/ConcurrentSolve.hpp) runs several LP
// engines on the same model at once and keeps whichever finishes first. The
// losers then have to be STOPPED -- otherwise the solve takes as long as the
// slowest engine and the whole exercise is pointless.
//
// There is no portable way to kill a thread safely, so cancellation is
// cooperative: the driver sets the flag, and each engine reads it once per
// iteration and returns early. One relaxed atomic load per simplex pivot or
// per PDLP iteration is far below the noise floor of the iteration itself.
//
// `memory_order_relaxed` is deliberate and sufficient. The flag carries no
// data and guards nothing: the only requirement is that the store eventually
// becomes visible, and the join that follows provides all the ordering the
// results need. An acquire/release pair here would cost a fence per iteration
// to establish a happens-before relationship nothing uses.
//
// An engine that stops for this reason reports `NotConverged` -- "still
// running, or stalled without reliable classification" (core/Types.hpp) --
// which is exactly true of an engine that was interrupted. It must never
// report `Optimal`, because it did not prove optimality.

#ifndef SOVSOLVE_CORE_CANCEL_HPP
#define SOVSOLVE_CORE_CANCEL_HPP

#include <atomic>

namespace sovsolve::core {

class CancelToken {
 public:
  CancelToken() = default;
  CancelToken(const CancelToken&) = delete;
  CancelToken& operator=(const CancelToken&) = delete;

  void cancel() noexcept { flag_.store(true, std::memory_order_relaxed); }

  [[nodiscard]] bool cancelled() const noexcept {
    return flag_.load(std::memory_order_relaxed);
  }

 private:
  std::atomic<bool> flag_{false};
};

/// True when `token` is non-null and has been cancelled. The null check is
/// what makes a normal single-engine solve pay nothing for this.
[[nodiscard]] inline bool is_cancelled(const CancelToken* token) noexcept {
  return token != nullptr && token->cancelled();
}

}  // namespace sovsolve::core

#endif  // SOVSOLVE_CORE_CANCEL_HPP
