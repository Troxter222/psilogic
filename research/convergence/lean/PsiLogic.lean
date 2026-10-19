import Mathlib.Data.Real.Basic
import Mathlib.Algebra.BigOperators.Group.Finset.Basic
import Mathlib.Tactic.Linarith
import Mathlib.Tactic.FieldSimp
import Mathlib.Tactic.Ring
import Mathlib.Tactic.NormNum

/-!
Scalar algebraic core of the PsiLogic analysis. These statements formalize only
what their explicit hypotheses say. In particular, `scalar_descent` assumes the
smooth upper-bound inequality; it does not establish differentiability or derive
that inequality for the optimizer. No stochastic process is formalized here.
-/

open scoped BigOperators

namespace PsiLogic

/-- The implemented upper-only damping clamp, on its intended nonnegative domain. -/
noncomputable def trust (z c : ℝ) : ℝ := 1 - min z (min c (99 / 100))

/-- A nonnegative damping input and cap yield the exact cap-dependent bounds. -/
theorem trust_bounds (z c : ℝ) (hz : 0 ≤ z) (hc : 0 ≤ c) :
    1 - min c (99 / 100) ≤ trust z c ∧ trust z c ≤ 1 := by
  have hcap : 0 ≤ min c (99 / 100 : ℝ) := le_min hc (by norm_num)
  have hmin : 0 ≤ min z (min c (99 / 100 : ℝ)) := le_min hz hcap
  have hupper := min_le_right z (min c (99 / 100 : ℝ))
  unfold trust
  constructor <;> linarith

/-- The internal 0.99 cap gives a strictly positive 0.01 floor. -/
theorem trust_floor (z c : ℝ) : (1 / 100 : ℝ) ≤ trust z c := by
  have h₁ := min_le_right z (min c (99 / 100 : ℝ))
  have h₂ := min_le_right c (99 / 100 : ℝ)
  unfold trust
  linarith

/-- Normalization uses the *updated* norm EMA, which bounds the ratio by 100. -/
theorem normalized_ema_bound (A r eps : ℝ)
    (hA : 0 ≤ A) (hr : 0 ≤ r) (heps : 0 < eps) :
    0 ≤ r / ((99 / 100) * A + (1 / 100) * r + eps) ∧
    r / ((99 / 100) * A + (1 / 100) * r + eps) ≤ 100 := by
  have hd : 0 < (99 / 100 : ℝ) * A + (1 / 100) * r + eps := by
    nlinarith
  constructor
  · exact div_nonneg hr (le_of_lt hd)
  · apply (div_le_iff₀ hd).2
    nlinarith

/-- Both fast and slow EMA recurrences preserve the interval [0,100]. -/
theorem ema_interval (b x z : ℝ)
    (hb0 : 0 ≤ b) (hb1 : b ≤ 1)
    (hx0 : 0 ≤ x) (hx100 : x ≤ 100)
    (hz0 : 0 ≤ z) (hz100 : z ≤ 100) :
    0 ≤ b * x + (1 - b) * z ∧ b * x + (1 - b) * z ≤ 100 := by
  have hp₁ : 0 ≤ b * x := mul_nonneg hb0 hx0
  have hp₂ : 0 ≤ (1 - b) * z := mul_nonneg (sub_nonneg.mpr hb1) hz0
  have hu₁ : b * x ≤ b * 100 := mul_le_mul_of_nonneg_left hx100 hb0
  have hu₂ : (1 - b) * z ≤ (1 - b) * 100 :=
    mul_le_mul_of_nonneg_left hz100 (sub_nonneg.mpr hb1)
  constructor <;> nlinarith

/-- Bias correction also rescales epsilon in the corrected denominator.
Here q is sqrt(1-beta2^t), a is 1-beta1^t, and r is sqrt(v_t). -/
theorem bias_epsilon_identity (m q a r eps : ℝ)
    (_hq : q ≠ 0) (ha : a ≠ 0) (hd : r + eps ≠ 0) :
    (q / a) * (m / (r + eps)) =
      (m / a) / (r / q + eps / q) := by
  rw [← add_div]
  field_simp [_hq, ha, hd]
  ring

/-- Multiplying an update by trust equals dividing its denominator by trust. -/
theorem trust_denominator_identity (m d s : ℝ) (hd : d ≠ 0) (_hs : s ≠ 0) :
    m / (d / s) = s * (m / d) := by
  field_simp [hd, _hs]
  ring

/-- Scalar absorption of the smoothness error under the explicit step restriction. -/
theorem scalar_descent (Fnext F alpha a L M G2 : ℝ)
    (halpha : 0 ≤ alpha) (hG : 0 ≤ G2)
    (hstep : L * alpha * M ^ 2 ≤ a)
    (hsmooth : Fnext ≤ F - alpha * a * G2 + L * alpha ^ 2 * M ^ 2 * G2 / 2) :
    Fnext ≤ F - alpha * a * G2 / 2 := by
  have hprod := mul_le_mul_of_nonneg_right hstep (mul_nonneg halpha hG)
  nlinarith

/-- Explicit finite telescoping. The descent premise is an assumption. -/
theorem finite_descent_sum (F w : ℕ → ℝ) (N : ℕ)
    (h : ∀ t < N, F (t + 1) ≤ F t - w t) :
    ∑ t ∈ Finset.range N, w t ≤ F 0 - F N := by
  induction N with
  | zero => simp
  | succ N ih =>
      have hold : ∀ t < N, F (t + 1) ≤ F t - w t := by
        intro t ht
        exact h t (Nat.lt_trans ht (Nat.lt_succ_self N))
      have hprev := ih hold
      have hlast := h N (Nat.lt_succ_self N)
      rw [Finset.sum_range_succ]
      linarith

/-- A lower bound on the last objective value yields the weighted certificate. -/
theorem weighted_stationarity_certificate (F w : ℕ → ℝ) (N : ℕ) (Flow : ℝ)
    (h : ∀ t < N, F (t + 1) ≤ F t - w t) (hlower : Flow ≤ F N) :
    ∑ t ∈ Finset.range N, w t ≤ F 0 - Flow := by
  have ht := finite_descent_sum F w N h
  linarith

/-- Finite 1/N stationarity certificate for a constant positive descent coefficient.
The application must separately establish that G2 is the squared gradient norm
and that the displayed descent premise holds. -/
theorem average_stationarity_bound (F G2 : ℕ → ℝ) (N : ℕ)
    (Flow alpha a : ℝ) (hN : 0 < N) (halpha : 0 < alpha) (ha : 0 < a)
    (h : ∀ t < N, F (t + 1) ≤ F t - alpha * a * G2 t / 2)
    (hlower : Flow ≤ F N) :
    (∑ t ∈ Finset.range N, G2 t) / (N : ℝ) ≤
      2 * (F 0 - Flow) / (alpha * a * (N : ℝ)) := by
  have hw := weighted_stationarity_certificate F
    (fun t => alpha * a * G2 t / 2) N Flow h hlower
  have hfactor : ∀ k : ℕ, (∑ t ∈ Finset.range k, alpha * a * G2 t / 2) =
      (alpha * a / 2) * (∑ t ∈ Finset.range k, G2 t) := by
    intro k
    induction k with
    | zero => simp
    | succ k ih =>
        simp only [Finset.sum_range_succ]
        rw [ih]
        ring
  rw [hfactor N] at hw
  have hNr : (0 : ℝ) < (N : ℝ) := Nat.cast_pos.mpr hN
  have hd : 0 < alpha * a * (N : ℝ) := mul_pos (mul_pos halpha ha) hNr
  apply (div_le_div_iff₀ hNr hd).2
  have hmul := mul_le_mul_of_nonneg_right hw (le_of_lt hNr)
  nlinarith

end PsiLogic

#print axioms PsiLogic.trust_bounds
#print axioms PsiLogic.trust_floor
#print axioms PsiLogic.normalized_ema_bound
#print axioms PsiLogic.ema_interval
#print axioms PsiLogic.bias_epsilon_identity
#print axioms PsiLogic.trust_denominator_identity
#print axioms PsiLogic.scalar_descent
#print axioms PsiLogic.finite_descent_sum
#print axioms PsiLogic.weighted_stationarity_certificate
#print axioms PsiLogic.average_stationarity_bound
