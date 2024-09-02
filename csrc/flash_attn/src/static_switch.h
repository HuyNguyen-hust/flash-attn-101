#pragma once

#define BOOL_SWITCH(COND, CONST_NAME, ...)      \
  [&] {                                         \
    if (COND) {                                 \
      constexpr static bool CONST_NAME = true;  \
      return __VA_ARGS__();                     \
    } else {                                    \
      constexpr static bool CONST_NAME = false; \
      return __VA_ARGS__();                     \
    }                                           \
  }()

// BOOL_SWITCH is a function that takes a condition and a const name, e.g., Is_causal, and ...(or __VA_ARGS__)
// Depending on the condition, the const name is either true or false
// Finally it call the __VA_ARGS__ function

#define FP16_SWITCH(COND, ...)               \
  [&] {                                      \
    if (COND) {                              \
      using elem_type = cutlass::half_t;     \
      return __VA_ARGS__();                  \
    } else {                                 \
      using elem_type = cutlass::bfloat16_t; \
      return __VA_ARGS__();                  \
    }                                        \
  }()

// FP16_SWITCH is a function that takes a condition, and ...
// Depending on the condition, the elem_type is either cutlass::half_t or cutlass::bfloat16_t
// Finally it call the __VA_ARGS__ function

#define HEADDIM_SWITCH(HEADDIM, ...)       \
  [&] {                                    \
    if (HEADDIM <= 32) {                   \
      constexpr static int kHeadDim = 32;  \
      return __VA_ARGS__();                \
    } else if (HEADDIM <= 64) {            \
      constexpr static int kHeadDim = 64;  \
      return __VA_ARGS__();                \
    }                                      \
  }()

// HEADDIM_SWITCH is a function that takes a head dimension, and ...
// Depending on the head dimension, the kHeadDim is either 32, 64, 96, 128, 160, 192, or 256
// Finally it call the __VA_ARGS__ function