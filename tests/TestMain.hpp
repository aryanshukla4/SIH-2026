// Minimal test harness.
//
// Deliberately dependency-free rather than GoogleTest or Catch2. The project
// carries a "shall not be built upon any existing open source solver library"
// constraint; a test framework is plainly not a solver library, but keeping
// third_party/ down to zlib alone means the question never has to be argued.
// It is also about sixty lines, against a multi-megabyte dependency.

#ifndef SOVSOLVE_TESTS_TEST_MAIN_HPP
#define SOVSOLVE_TESTS_TEST_MAIN_HPP

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <string_view>
#include <vector>

namespace sovsolve::test {

struct Failure {
  std::string file;
  int line;
  std::string expression;
  std::string detail;
};

inline std::vector<Failure>& failures() {
  static std::vector<Failure> f;
  return f;
}

inline int& checks_run() {
  static int n = 0;
  return n;
}

inline void record(const char* file, int line, const char* expr,
                   std::string detail) {
  failures().push_back(Failure{file, line, expr, std::move(detail)});
}

inline bool close(double a, double b, double tol) {
  const double diff = std::fabs(a - b);
  if (diff <= tol) return true;
  const double scale = std::fmax(std::fabs(a), std::fabs(b));
  return diff <= tol * scale;
}

inline int report(std::string_view suite) {
  const auto& f = failures();
  if (f.empty()) {
    std::printf("PASS  %.*s  (%d checks)\n", static_cast<int>(suite.size()),
                suite.data(), checks_run());
    return 0;
  }
  std::printf("FAIL  %.*s  (%zu of %d checks failed)\n",
              static_cast<int>(suite.size()), suite.data(), f.size(),
              checks_run());
  for (const auto& x : f) {
    std::printf("  %s:%d\n    %s\n", x.file.c_str(), x.line,
                x.expression.c_str());
    if (!x.detail.empty()) std::printf("    %s\n", x.detail.c_str());
  }
  return 1;
}

}  // namespace sovsolve::test

#define CHECK(expr)                                                       \
  do {                                                                    \
    ++::sovsolve::test::checks_run();                                     \
    if (!(expr)) ::sovsolve::test::record(__FILE__, __LINE__, #expr, ""); \
  } while (0)

#define CHECK_EQ(a, b)                                                       \
  do {                                                                       \
    ++::sovsolve::test::checks_run();                                        \
    const auto va_ = (a);                                                    \
    const auto vb_ = (b);                                                    \
    if (!(va_ == vb_)) {                                                     \
      ::sovsolve::test::record(__FILE__, __LINE__, #a " == " #b,             \
                               "got " + std::to_string(va_) + ", expected " + \
                                   std::to_string(vb_));                     \
    }                                                                        \
  } while (0)

#define CHECK_NEAR(a, b, tol)                                                \
  do {                                                                       \
    ++::sovsolve::test::checks_run();                                        \
    const double va_ = static_cast<double>(a);                               \
    const double vb_ = static_cast<double>(b);                               \
    if (!::sovsolve::test::close(va_, vb_, (tol))) {                         \
      ::sovsolve::test::record(__FILE__, __LINE__, #a " ~= " #b,             \
                               "got " + std::to_string(va_) + ", expected " + \
                                   std::to_string(vb_));                     \
    }                                                                        \
  } while (0)

#endif  // SOVSOLVE_TESTS_TEST_MAIN_HPP
