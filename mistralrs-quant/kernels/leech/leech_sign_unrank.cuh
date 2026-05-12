// Algebraic even-class sign-bitmask unrank — replaces `valid_signs_flat[s_lo + s_idx]`.
//
// Mathematical derivation (proven bit-equal to legacy table across 381M pairs at
// ms_max=18 — see antsquant/tools/prototype_sign_unrank.py and
// packer/core/sign_unrank.py):
//
//   Niemeier Λ24 even-class constraint:  Σ ε_i · v_i ≡ 0 (mod 8),  ε_i = 2 b_i − 1.
//   Since every leader value is even (v_i mod 4 ∈ {0, 2}), this reduces to ONE
//   GF(2) parity equation:
//
//     popcount(b ∧ V2_mask) ≡ T (mod 2),    where
//       V2_mask = { i : v_i ≡ 2 (mod 4) },
//       T       = (Σ v_i) / 4  mod 2
//
// Per-class metadata: V2_mask:uint32, dep_bit:int8 (-1 ⇒ |V_2|=0 identity case),
// T:uint8 — total 8 B/class, ~10 KB at ms_max=18.
//
// Cost: 6 PTX ops on the |V_2| > 0 path, 0 on the identity path. Branchless
// inside the dep_bit ≥ 0 path; the dep_bit < 0 branch is class-uniform.

#pragma once
#include <cstdint>

namespace leech {

// Returns the sign bitmask for `s_idx ∈ [0, two_B[g])`, bit-identical to
// `valid_signs_flat[signs_ofs[g] + s_idx]` in the legacy table.
//
// `dep_bit < 0` (encoded as int8 = -1) flags the |V_2| = 0 case where every
// bit of `s_idx` is a free choice — the bitmask is just s_idx itself.
__device__ __forceinline__ uint64_t even_sign_unrank(
    uint64_t s_idx,
    uint32_t V2_mask,
    int      dep_bit,
    uint32_t T
) {
    if (dep_bit < 0) {
        // |V_2| = 0 — identity unrank.
        return s_idx;
    }
    // Spread the (n-1)-bit s_idx around dep_bit:
    //   low (dep_bit) bits stay; remaining bits shift left by 1.
    uint64_t lo_mask = (1ull << dep_bit) - 1ull;
    uint64_t p = (s_idx & lo_mask) | ((s_idx >> dep_bit) << (dep_bit + 1));

    // popcount-parity of (p ∧ V2_mask). Use __popcll → single PTX popc.b64.
    uint64_t cur_parity = static_cast<uint64_t>(__popcll(p & static_cast<uint64_t>(V2_mask))) & 1ull;

    // Force parity to T by flipping dep_bit if needed.
    p |= ((cur_parity ^ static_cast<uint64_t>(T)) & 1ull) << dep_bit;
    return p;
}

}  // namespace leech
